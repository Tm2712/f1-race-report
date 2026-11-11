"""Pit-lap what-if model, from OpenF1 clean laps, stints and lap end times.

Idea: fit a simple pace model to every clean lap, then re-time the two stints around a pit stop as if
the stop had happened earlier or later, keep everyone else's actual times, and re-rank the finish.
Pure pandas/numpy, so it can be tested without network or plotting libraries.
"""
import html

import numpy as np
import pandas as pd

from insights import _ord

MAX_SHIFT = 8            # laps earlier or later to test
MIN_STINT = 5            # shortest stint we will propose
MIN_ACTUAL_STINT = 8     # only analyse stops between two real stints (skips pit-lane starts, early incidents)
MIN_COMPOUND_LAPS = 30   # clean laps a compound needs before we trust its wear rate
BOOTSTRAP = 100
MIN_GAIN = 1.0           # seconds: smaller modelled gains are not worth reporting


class PaceModel:
    """fuel-corrected lap time = driver pace + compound offset + wear rate per compound x tyre age + lap trend."""

    def __init__(self, df):
        df = df.dropna(subset=["tc", "TyreLife", "compound"]).copy()
        df["compound"] = df["compound"].astype(str).str.upper()
        self.drivers = sorted(df["driver_number"].unique())
        self.comps = sorted(df["compound"].unique())
        self.di = {d: i for i, d in enumerate(self.drivers)}
        self.ci = {c: i for i, c in enumerate(self.comps)}
        self.nd, self.nc = len(self.drivers), len(self.comps)
        self.n = self.nd + (self.nc - 1) + self.nc + 1
        X = np.array([self.x(r.driver_number, r.compound, r.TyreLife, r.lap_number) for r in df.itertuples()])
        y = df["tc"].to_numpy(dtype=float)
        self.coef = np.linalg.lstsq(X, y, rcond=None)[0]
        rng = np.random.default_rng(0)
        draws = (rng.integers(0, len(y), len(y)) for _ in range(BOOTSTRAP))
        self.boot = np.array([np.linalg.lstsq(X[i], y[i], rcond=None)[0] for i in draws])
        self.max_age = df.groupby("compound")["TyreLife"].max().to_dict()
        self.n_laps = df.groupby("compound").size().to_dict()

    def x(self, drv, comp, age, lap=0.0):
        v = np.zeros(self.n)
        v[self.di[drv]] = 1.0
        c = self.ci[comp]
        if c > 0:
            v[self.nd + c - 1] = 1.0
        v[self.nd + self.nc - 1 + c] = age
        v[-1] = lap
        return v

    def usable(self, comp):
        return comp in self.ci and self.n_laps.get(comp, 0) >= MIN_COMPOUND_LAPS


def _stints(g, last):
    return [dict(c=str(r.compound).upper(), a=int(r.lap_start),
                 b=int(r.lap_end) if pd.notna(r.lap_end) else last,
                 age0=int(r.tyre_age_at_start) if pd.notna(r.tyre_age_at_start) else 0)
            for r in g.sort_values("stint_number").itertuples()]


def sweep(model, stints, et, neutral=frozenset(), exclude=frozenset()):
    """Modelled change in finish time (delta, seconds; negative = faster) for every stop moved by -8..+8 laps."""
    fin = et.sort_values("lap_number").groupby("driver_number").tail(1).set_index("driver_number")
    rows = []
    for d, g in stints.groupby("driver_number"):
        if d not in model.di or d not in fin.index or d in exclude:  # exclude: drivers who did not finish
            continue
        mine, t_fin = int(fin.loc[d, "lap_number"]), fin.loc[d, "t"]
        others = fin.drop(index=d)

        def pos(tt):
            return 1 + int(((others["lap_number"] > mine) | ((others["lap_number"] == mine) & (others["t"] < tt))).sum())

        st = _stints(g, mine)
        for k in range(len(st) - 1):
            A, B = st[k], st[k + 1]
            if not (model.usable(A["c"]) and model.usable(B["c"])):
                continue
            if A["b"] - A["a"] + 1 < MIN_ACTUAL_STINT or B["b"] - B["a"] + 1 < MIN_ACTUAL_STINT:
                continue
            stop = A["b"]
            for s in range(-MAX_SHIFT, MAX_SHIFT + 1):
                new = stop + s
                if s == 0 or new - A["a"] + 1 < MIN_STINT or B["b"] - new < MIN_STINT:
                    continue
                if A["age0"] + new - A["a"] + 1 > model.max_age[A["c"]] or B["age0"] + B["b"] - new > model.max_age[B["c"]]:
                    continue  # would need tyre ages never seen in this race
                if any(lap in neutral for lap in range(min(stop, new) - 2, max(stop, new) + 3)):
                    continue  # stops near a safety car are a different problem
                S = np.zeros(model.n)
                for lap in range(A["a"], B["b"] + 1):
                    old = (A["c"], A["age0"] + lap - A["a"] + 1) if lap <= stop else (B["c"], B["age0"] + lap - stop)
                    cur = (A["c"], A["age0"] + lap - A["a"] + 1) if lap <= new else (B["c"], B["age0"] + lap - new)
                    S += model.x(d, *cur) - model.x(d, *old)
                delta = float(model.coef @ S)
                rows.append({"driver_number": d, "stop": k + 1, "stop_lap": stop, "shift": s, "new_lap": new,
                             "delta": delta, "se": float((model.boot @ S).std()),
                             "pos_actual": pos(t_fin), "pos_sim": pos(t_fin + delta)})
    return pd.DataFrame(rows)


def best_per_driver(sw):
    if sw.empty:
        return sw
    b = sw.loc[sw.groupby("driver_number")["delta"].idxmin()].copy()
    b["gain"] = -b["delta"]
    return b.reset_index(drop=True)


def reliable(sw):
    b = best_per_driver(sw)
    if b.empty:
        return b
    return b[(b["gain"] >= MIN_GAIN) & (b["gain"] > 2 * b["se"])].sort_values("gain", ascending=False)


def write_whatif(sw, names):
    if sw.empty:
        return ["Not enough clean laps or tyre data to run the model for this race."]
    good = reliable(sw)
    if good.empty:
        return ["For every driver, no alternative stop lap was modelled as clearly faster than the one they used "
                "(any gain was within the model's noise)."]
    out = [f"{len(good)} of {sw['driver_number'].nunique()} drivers had a modelled gain of at least {MIN_GAIN:.0f} s "
           "from a different stop lap. The largest:"]
    for r in good.head(3).itertuples():
        who = names.get(r.driver_number, r.driver_number)
        when = "later" if r.shift > 0 else "earlier"
        move = (f"enough to move from {_ord(r.pos_actual)} to {_ord(r.pos_sim)}" if r.pos_sim < r.pos_actual
                else "not enough to change their finishing position")
        edge = " The best option is at the edge of the range tested, so treat this with caution." if abs(r.shift) == MAX_SHIFT else ""
        out.append(f"{who}: stopping on lap {r.new_lap} instead of {r.stop_lap} ({abs(r.shift)} laps {when}) "
                   f"is modelled {r.gain:.1f} s (±{r.se:.1f}) faster, {move}.{edge}")
    return out


METHOD = (
    "<details><summary>How the what-if is calculated</summary><p>A pace model is fitted to every clean lap: "
    "fuel-corrected lap time = driver pace + compound offset + wear rate per compound x tyre age + a race-lap trend. "
    "For each stop, the model re-times that driver's two stints as if the stop happened up to 8 laps earlier or later, "
    "keeps every other driver's actual times, and re-ranks the finish by timing data (which can differ from the "
    "official result after penalties). The ± comes from refitting on 100 resampled sets of laps; it understates the "
    "true uncertainty because laps are not independent. Picking the best of up to 16 shifts flatters the result, so "
    "read gains as upper estimates. The model ignores traffic, rivals reacting, tyre cliffs and any tyre age not seen "
    "in the race (those shifts are skipped), stops near safety cars are excluded, as are drivers who did not finish and stops that follow or precede a stint shorter than 8 laps (such as pit-lane starts).</p></details>")


def whatif_html(sw, names):
    items = "".join(f"<li>{html.escape(s)}</li>" for s in write_whatif(sw, names))
    return f"<section><h2>Pit-lap what-if</h2><ul>{items}</ul>{METHOD}</section>"
