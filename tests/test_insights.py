"""Run with:  python tests/test_insights.py   (needs pandas only)."""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
import insights  # noqa: E402

T0 = pd.Timestamp("2025-03-16T04:00:00Z")


def fake_race():
    """VER leads, NOR is 2 s behind and pits first (lap 20), VER responds on lap 24. LEC is far back."""
    cfg = {1: (0.0, 24), 4: (2.0, 20), 16: (10.0, 30)}
    laps, stints = [], []
    for n, (offset, stop) in cfg.items():
        t = offset
        for lap in range(1, 51):
            age = lap if lap <= stop else lap - stop
            dur = 90 + 0.1 * age + (21 if lap == stop + 1 else 0)
            laps.append(dict(driver_number=n, lap_number=lap, lap_duration=dur, is_pit_out_lap=lap == stop + 1,
                             date_start=(T0 + pd.Timedelta(seconds=t)).isoformat()))
            t += dur
        stints += [dict(driver_number=n, stint_number=1, lap_start=1, lap_end=stop),
                   dict(driver_number=n, stint_number=2, lap_start=stop + 1, lap_end=50)]
    return pd.DataFrame(laps), pd.DataFrame(stints)


def test_undercut_detected():
    laps, stints = fake_race()
    uc = insights.undercut_table(insights.end_times(laps), stints)
    assert len(uc) == 1, uc
    row = uc.iloc[0]
    assert (row.attacker, row.ahead, row.lap, row.reply_lap) == (4, 1, 20, 24)
    assert row.worked and abs(row.margin - 5.2) < 0.2, row.margin
    assert abs(row.gap - 2.0) < 0.1


def test_safety_car_stops_ignored():
    laps, stints = fake_race()
    uc = insights.undercut_table(insights.end_times(laps), stints, neutral={20})
    assert uc.empty


def test_pace_vs_result_and_text():
    names = {1: "VER", 4: "NOR", 16: "LEC", 44: "HAM"}
    med = pd.Series({"NOR": 90.0, "VER": 90.1, "LEC": 91.0, "HAM": 92.0})
    result = pd.DataFrame([dict(driver_number=n, position=p, dnf=False, dns=False, dsq=False)
                           for n, p in [(1, 1), (16, 2), (44, 3), (4, 4)]])
    pace = insights.pace_vs_result(med, result, names)
    text = " ".join(insights.write_findings(pd.DataFrame(), pace, names))
    assert "NOR had the 1st-best" in text and "finished 4th" in text, text
    assert "VER had" not in text


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("passed", name)
