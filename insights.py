"""Race findings: pit-stop (undercut) analysis and pace-vs-result, from OpenF1 laps and stints.

Pure pandas, no plotting, so it can be tested without any network or chart library.
"""
import html

import pandas as pd

UNDERCUT_GAP = 3.0      # seconds: attacker must be this close behind the car ahead before stopping
RESPONSE_WINDOW = 10    # laps the car ahead has to respond before we call it "stayed out"
PACE_MISMATCH = 3       # places between pace rank and finish before we mention it


def end_times(laps):
    """Seconds since the first lap start at which each driver completed each lap.

    Uses the start time of the next lap as the finish time of this one (OpenF1 calls these
    timestamps approximate, so gaps are good to roughly a few tenths of a second).
    """
    l = laps.dropna(subset=["date_start"]).copy()
    l["start"] = pd.to_datetime(l["date_start"], utc=True, format="ISO8601")
    l = l.sort_values(["driver_number", "lap_number"])
    g = l.groupby("driver_number")
    l["end"] = g["start"].shift(-1)
    consecutive = g["lap_number"].shift(-1) == l["lap_number"] + 1
    l.loc[~consecutive, "end"] = pd.NaT
    last = l["end"].isna() & l["lap_duration"].notna() & g["lap_number"].shift(-1).isna()
    l.loc[last, "end"] = l.loc[last, "start"] + pd.to_timedelta(l.loc[last, "lap_duration"], unit="s")
    l = l.dropna(subset=["end"])
    l["t"] = (l["end"] - l["start"].min()).dt.total_seconds()
    l["rank"] = l.groupby("lap_number")["t"].rank(method="first")
    return l[["driver_number", "lap_number", "t", "rank"]]


def stop_laps(stints):
    """Pit-in lap of every stop, per driver (the last lap of every stint except the final one)."""
    out = {}
    for d, g in stints.sort_values("stint_number").groupby("driver_number"):
        out[d] = [int(x) for x in g["lap_end"].iloc[:-1].dropna()]
    return out


def undercut_table(et, stints, neutral=frozenset()):
    """Every green-flag stop made by a car within UNDERCUT_GAP seconds behind another car.

    The outcome is judged two laps after the car ahead responds (or RESPONSE_WINDOW laps later
    if it stayed out): did the stopper end up in front?
    """
    stops = stop_laps(stints)
    at = et.set_index(["lap_number", "driver_number"])
    rows = []
    for a, a_stops in stops.items():
        for lap in a_stops:
            if lap in neutral or (lap - 1, a) not in at.index:
                continue
            me = at.loc[(lap - 1, a)]
            ahead = et[(et["lap_number"] == lap - 1) & (et["rank"] == me["rank"] - 1)]
            if ahead.empty:
                continue
            b, gap = int(ahead.iloc[0]["driver_number"]), me["t"] - ahead.iloc[0]["t"]
            if gap > UNDERCUT_GAP or lap in stops.get(b, []):
                continue
            last_stop = lambda d: max([x for x in stops.get(d, []) if x < lap], default=0)
            if last_stop(b) > last_stop(a):  # car ahead is already on fresher tyres: not an undercut
                continue
            reply = [s for s in stops.get(b, []) if lap < s <= lap + RESPONSE_WINDOW]
            b_stop = reply[0] if reply else None
            if b_stop in neutral:
                continue
            check = (b_stop + 2) if b_stop else lap + RESPONSE_WINDOW
            if (check, a) not in at.index or (check, b) not in at.index:
                continue
            margin = at.loc[(check, b), "t"] - at.loc[(check, a), "t"]  # >0: attacker now ahead
            rows.append({"attacker": a, "ahead": b, "lap": lap, "gap": gap, "reply_lap": b_stop,
                         "check_lap": check, "margin": margin, "worked": margin > 0})
    return pd.DataFrame(rows)


def pace_vs_result(med, result, names):
    """Compare each classified finisher's median clean-lap pace rank with their finishing position."""
    r = result.copy()
    for flag in ("dnf", "dns", "dsq"):
        if flag in r:
            r = r[~r[flag].fillna(False).astype(bool)]
    r = r.dropna(subset=["position"])
    r["abbr"] = r["driver_number"].map(names)
    r = r[r["abbr"].isin(med.index)].copy()
    r["pace_rank"] = r["abbr"].map(med).rank(method="first")
    r["diff"] = r["pace_rank"] - r["position"]  # >0: finished ahead of what pace suggests
    return r[["abbr", "pace_rank", "position", "diff"]]


def _ord(n):
    n = int(n)
    return f"{n}{'th' if 10 <= n % 100 <= 20 else {1: 'st', 2: 'nd', 3: 'rd'}.get(n % 10, 'th')}"


def write_findings(uc, pace, names):
    out = []
    if uc.empty:
        out.append(f"No clean undercut attempts found: no car stopped under green-flag running while "
                   f"within {UNDERCUT_GAP:.0f} s behind another.")
    else:
        won = uc[uc["worked"]]
        out.append(f"{len(uc)} stop(s) were made by a car within {UNDERCUT_GAP:.0f} s behind another; "
                   f"{len(won)} ended up in front, {len(uc) - len(won)} did not.")
        for label, rows, ascending in (("Best", won, False), ("Worst", uc[~uc["worked"]], True)):
            if rows.empty:
                continue
            r = rows.sort_values("margin", ascending=ascending).iloc[0]
            a, b = names.get(r["attacker"], r["attacker"]), names.get(r["ahead"], r["ahead"])
            reply = f"{b} responded on lap {int(r['reply_lap'])}" if pd.notna(r["reply_lap"]) else f"{b} stayed out"
            side = f"was {r['margin']:.1f} s ahead" if r["worked"] else f"was still {-r['margin']:.1f} s behind"
            out.append(f"{label} result: {a} stopped on lap {int(r['lap'])}, {r['gap']:.1f} s behind {b} ({reply}), "
                       f"and {side} by lap {int(r['check_lap'])}.")
    for sign, text in ((1, "finished ahead of"), (-1, "finished behind")):
        d = pace[pace["diff"] * sign >= PACE_MISMATCH].sort_values("diff", ascending=sign < 0)
        if not d.empty:
            r = d.iloc[0]
            out.append(f"{r['abbr']} had the {_ord(r['pace_rank'])}-best median clean-lap pace but finished "
                       f"{_ord(r['position'])}: {text} their pace. Check for penalties, damage, safety cars or strategy.")
    return out


METHOD = (
    "<details><summary>How these findings are calculated</summary>"
    "<p>Gaps and running order come from OpenF1 lap start times, which the API describes as approximate, "
    "so treat gaps as accurate to a few tenths of a second. A stop counts as an undercut attempt when the car "
    "was within 3 s behind another car that had not pitted more recently, and neither stop happened under a safety car. The result is judged two laps "
    "after the car ahead pitted, or 10 laps later if it stayed out. The analysis does not know tyre age differences, "
    "traffic on rejoining or team intent. Pace uses green-flag laps with in, out and first laps removed and a rough "
    "fuel correction of 0.03 s per lap, so small pace gaps are not meaningful.</p></details>")


def findings_html(uc, pace, names):
    items = "".join(f"<li>{html.escape(s)}</li>" for s in write_findings(uc, pace, names))
    table = ""
    if not uc.empty:
        body = "".join(
            f"<tr><td>{html.escape(str(names.get(r.attacker, r.attacker)))}</td>"
            f"<td>{html.escape(str(names.get(r.ahead, r.ahead)))}</td><td>{int(r.lap)}</td><td>{r.gap:.1f} s</td>"
            f"<td>{'stayed out' if pd.isna(r.reply_lap) else int(r.reply_lap)}</td>"
            f"<td>{'Gained place' if r.worked else 'Still behind'} ({r.margin:+.1f} s)</td></tr>"
            for r in uc.sort_values("lap").itertuples())
        table = ("<div style='overflow-x:auto'><table><tr><th>Stopped</th><th>Car ahead</th><th>On lap</th>"
                 f"<th>Gap before</th><th>Car ahead stopped</th><th>Result</th></tr>{body}</table></div>")
    return f"<section><h2>Race findings</h2><ul>{items}</ul>{table}{METHOD}</section>"
