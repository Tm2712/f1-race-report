"""Run with:  python tests/test_whatif.py   (needs pandas and numpy only)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
import whatif  # noqa: E402

CFG = {1: (0.0, 25), 4: (3.0, 25), 16: (6.0, 15), 44: (9.0, 24)}  # driver: (start offset s, stop lap)


def fake_race(seed=0):
    """One compound, wear of 0.1 s per lap of tyre age, 21 s pit cost. Driver 16 stops far too early."""
    rng = np.random.default_rng(seed)
    clean, stints, et = [], [], []
    for n, (offset, stop) in CFG.items():
        t = offset
        for lap in range(1, 51):
            age = lap if lap <= stop else lap - stop
            dur = 90 + 0.1 * age + (21 if lap in (stop, stop + 1) and lap == stop + 1 else 0) + rng.normal(0, 0.05)
            t += dur
            et.append(dict(driver_number=n, lap_number=lap, t=t, rank=0))
            if lap > 1 and lap not in (stop, stop + 1):
                clean.append(dict(driver_number=n, lap_number=lap, tc=dur, compound="MEDIUM", TyreLife=age))
        stints += [dict(driver_number=n, stint_number=1, compound="MEDIUM", lap_start=1, lap_end=stop, tyre_age_at_start=0),
                   dict(driver_number=n, stint_number=2, compound="MEDIUM", lap_start=stop + 1, lap_end=50, tyre_age_at_start=0)]
    return pd.DataFrame(clean), pd.DataFrame(stints), pd.DataFrame(et)


def test_early_stopper_gains_from_stopping_later():
    clean, stints, et = fake_race()
    sw = whatif.sweep(whatif.PaceModel(clean), stints, et)
    best = whatif.best_per_driver(sw).set_index("driver_number")
    r = best.loc[16]
    assert r["shift"] == whatif.MAX_SHIFT and abs(r["gain"] - 9.6) < 1.0, r
    assert (r["pos_actual"], r["pos_sim"]) == (4, 3), r
    assert 16 in whatif.reliable(sw)["driver_number"].values


def test_balanced_stopper_has_no_reliable_gain():
    clean, stints, et = fake_race()
    good = whatif.reliable(whatif.sweep(whatif.PaceModel(clean), stints, et))
    assert 1 not in good["driver_number"].values and 4 not in good["driver_number"].values


def test_safety_car_and_extrapolation_guards():
    clean, stints, et = fake_race()
    model = whatif.PaceModel(clean)
    sw = whatif.sweep(model, stints, et, neutral=set(range(10, 31)))
    assert 16 not in sw.get("driver_number", pd.Series(dtype=int)).values  # stop lap 15 is near the safety car
    full = whatif.sweep(model, stints, et)
    ages = full[full.driver_number == 1]
    assert (ages["shift"].abs() <= whatif.MAX_SHIFT).all() and not ages.empty


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("passed", name)
