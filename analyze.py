
"""Automated F1 post-race report (data from the free OpenF1 API, no login needed).

Usage:
    python analyze.py              # latest completed race
    python analyze.py 2025 7       # a specific season and round
Outputs static pages into docs/ (served by GitHub Pages).
"""
import html
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests

ROOT = Path(__file__).parent
DOCS, RACES = ROOT / "docs", ROOT / "docs" / "races"
MANIFEST = DOCS / "races.json"
API = "https://api.openf1.org/v1"
COMPOUNDS = {"SOFT": "#E10600", "MEDIUM": "#F2B800", "HARD": "#8791A3",
             "INTERMEDIATE": "#2E9E4F", "WET": "#2F6FDE"}
FUEL_PER_LAP = 0.03  # seconds of lap time gained per lap of fuel burned (rough)
MIN_STINT_LAPS = 8
PLOTLY_JS = "https://cdn.plot.ly/plotly-2.35.2.min.js"


# ---------- data (OpenF1 allows 3 requests/second and 30/minute) ----------
def get(endpoint, **params):
    for attempt in range(6):
        r = requests.get(f"{API}/{endpoint}", params=params, timeout=90)
        if r.status_code == 429:  # rate limited: wait and retry
            time.sleep(15 * (attempt + 1))
            continue
        if r.status_code == 404:  # OpenF1 answers 404 when a query has no rows
            return pd.DataFrame()
        r.raise_for_status()
        time.sleep(2)
        return pd.DataFrame(r.json())
    raise SystemExit(f"OpenF1 kept rate-limiting the request for {endpoint}.")


def calendar(year):
    """All non-cancelled Grand Prix races of a season, with a round number."""
    s = get("sessions", year=year, session_name="Race")
    if s.empty:
        return s
    s = s[~s["is_cancelled"].fillna(False).astype(bool)].copy()
    s["start"] = pd.to_datetime(s["date_start"], utc=True)
    s["end"] = pd.to_datetime(s["date_end"], utc=True)
    s = s.sort_values("start").reset_index(drop=True)
    s["round"] = s.index + 1
    return s


def latest_race():
    now = pd.Timestamp.now(tz="UTC")
    for year in (now.year, now.year - 1):
        cal = calendar(year)
        done = cal[cal["end"] + pd.Timedelta(hours=1) < now] if not cal.empty else cal
        if not done.empty:
            return year, done.iloc[-1]
    raise SystemExit("No completed race found.")


# ---------- analysis ----------
def neutral_laps(rc):
    """Laps run under the safety car / VSC (plus the restart lap), from race control messages."""
    out, start = set(), None
    if rc.empty or "message" not in rc:
        return out
    for _, r in rc.sort_values("date").iterrows():
        msg, lap = str(r["message"]).upper(), r.get("lap_number")
        if pd.isna(lap):
            continue
        lap = int(lap)
        if "DEPLOYED" in msg and "SAFETY CAR" in msg and start is None:
            start = lap
        elif start is not None and ("IN THIS LAP" in msg or "ENDING" in msg):
            out.update(range(start, lap + 2))
            start = None
    return out


def clean_laps(laps, stints, rc):
    """Green-flag racing laps only: no lap 1, in/out laps, safety cars, or slow outliers."""
    pit_out = laps["is_pit_out_lap"].fillna(False).astype(bool)
    in_laps = set(zip(laps.loc[pit_out, "driver_number"], laps.loc[pit_out, "lap_number"] - 1))
    df = laps[~pit_out].dropna(subset=["lap_duration"])
    df = df[(df["lap_number"] > 1) & ~df["lap_number"].isin(neutral_laps(rc))]
    df = df[[(d, l) not in in_laps for d, l in zip(df["driver_number"], df["lap_number"])]]

    s = stints.copy()
    s["lap_end"] = s["lap_end"].fillna(laps["lap_number"].max())
    m = df.merge(s[["driver_number", "stint_number", "compound", "lap_start", "lap_end", "tyre_age_at_start"]],
                 on="driver_number")
    m = m[(m["lap_number"] >= m["lap_start"]) & (m["lap_number"] <= m["lap_end"])].copy()
    m["TyreLife"] = m["tyre_age_at_start"].fillna(0) + m["lap_number"] - m["lap_start"] + 1
    m = m[m["lap_duration"] <= 1.07 * m["lap_duration"].quantile(0.10)]
    m["tc"] = m["lap_duration"] - FUEL_PER_LAP * (laps["lap_number"].max() - m["lap_number"])  # fuel-corrected
    return m


def stint_degradation(df):
    rows = []
    for (drv, _), g in df.groupby(["Driver", "stint_number"]):
        comp = str(g["compound"].iloc[0]).upper()
        if len(g) < MIN_STINT_LAPS or comp not in COMPOUNDS:
            continue
        rows.append({"driver": drv, "compound": comp, "slope": np.polyfit(g["TyreLife"], g["tc"], 1)[0]})
    return pd.DataFrame(rows)


# ---------- charts ----------
def style(fig, height=520):
    fig.update_layout(template="simple_white", height=height, paper_bgcolor="rgba(0,0,0,0)",
                      plot_bgcolor="rgba(0,0,0,0)", font=dict(family="Barlow, sans-serif", size=13, color="#101828"),
                      margin=dict(l=50, r=20, t=20, b=50), legend=dict(orientation="h", y=-0.15))
    return fig


def chart_pace(df, colors):
    order = df.groupby("Driver")["tc"].median().sort_values().index
    fig = go.Figure()
    for d in order:
        fig.add_trace(go.Box(y=df.loc[df.Driver == d, "tc"], name=d, marker_color=colors.get(d, "#8791A3"),
                             boxpoints=False, showlegend=False))
    fig.update_yaxes(title="Fuel-corrected lap time (s)")
    return style(fig)


def chart_deg(deg):
    fig = go.Figure()
    for comp in [c for c in COMPOUNDS if c in set(deg["compound"])]:
        d = deg[deg.compound == comp]
        fig.add_trace(go.Box(y=d["slope"], name=comp.title(), marker_color=COMPOUNDS[comp], boxpoints="all",
                             jitter=0.4, pointpos=0, text=d["driver"], hovertemplate="%{text}: %{y:.3f} s/lap"))
    fig.add_hline(y=0, line_color="#101828", line_width=1)
    fig.update_yaxes(title="Lap time change per lap of tyre age (s)")
    return style(fig, 460)


def chart_strategy(stints, result, names):
    order = [names[n] for n in result.sort_values("position")["driver_number"] if n in names]
    st = stints.copy()
    st["Driver"] = st["driver_number"].map(names)
    st["compound"] = st["compound"].astype(str).str.upper()
    fig = go.Figure()
    for comp in [c for c in COMPOUNDS if c in set(st["compound"])]:
        d = st[st.compound == comp]
        fig.add_trace(go.Bar(y=d["Driver"], x=d["lap_end"] - d["lap_start"] + 1, base=d["lap_start"] - 1,
                             orientation="h", name=comp.title(),
                             marker=dict(color=COMPOUNDS[comp], line=dict(color="#fff", width=2)),
                             customdata=d["lap_end"], hovertemplate="%{y}: laps %{base}-%{customdata}<extra></extra>"))
    fig.update_yaxes(categoryorder="array", categoryarray=order, autorange="reversed")
    fig.update_xaxes(title="Lap")
    fig.update_layout(barmode="overlay")
    return style(fig, 620)


def chart_positions(result, grid, names):
    r = result.copy()
    for flag in ("dnf", "dns", "dsq"):
        r = r[~r[flag].fillna(False).astype(bool)] if flag in r else r
    r = r.dropna(subset=["position"]).sort_values("position")
    r["grid"] = r["driver_number"].map(grid.set_index("driver_number")["position"])
    r = r.dropna(subset=["grid"])
    if r.empty:
        return None
    r["gain"] = r["grid"] - r["position"]
    fig = go.Figure(go.Bar(x=r["driver_number"].map(names), y=r["gain"],
                           marker_color=np.where(r["gain"] >= 0, "#1F8A5B", "#E10600")))
    fig.update_yaxes(title="Places gained vs. grid")
    return style(fig, 400)


# ---------- page ----------
CSS = """
:root{--bg:#E9EDF2;--panel:#fff;--ink:#101828;--mute:#5B6677;--line:#CBD3DE;--red:#E10600}
@media (prefers-color-scheme:dark){:root{--bg:#161B24;--panel:#1F2631;--ink:#EDF0F5;--mute:#9AA5B6;--line:#334055}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:17px/1.55 Barlow,system-ui,sans-serif}
main{max-width:980px;margin:0 auto;padding:40px 20px 80px}
h1{font:700 clamp(44px,9vw,96px)/.92 "Barlow Condensed",sans-serif;margin:0 0 8px;letter-spacing:-.01em}
h1+p{color:var(--mute);margin:0 0 32px}
.facts{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:0;border-top:4px solid var(--red);margin-bottom:40px}
.facts div{padding:14px 16px 14px 0}.facts b{display:block;font:600 24px "Barlow Condensed",sans-serif}.facts span{color:var(--mute);font-size:15px}
section{background:var(--panel);border:1px solid var(--line);padding:24px;margin-bottom:24px}
h2{font:700 30px "Barlow Condensed",sans-serif;margin:0 0 4px}section p{color:var(--mute);margin:0 0 16px;max-width:70ch}
a{color:inherit}ul{list-style:none;padding:0}li{border-bottom:1px solid var(--line);padding:14px 0}li a{font:600 24px "Barlow Condensed",sans-serif;text-decoration:none}
li span{display:block;color:var(--mute);font-size:15px}footer{color:var(--mute);font-size:14px;margin-top:40px}
"""


def shell(title, body):
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>{html.escape(title)}</title>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Barlow:wght@400;600&family=Barlow+Condensed:wght@600;700&display=swap">
<script src="{PLOTLY_JS}"></script><style>{CSS}</style></head><body><main>{body}</main></body></html>"""


def build_race_page(meta, facts, charts):
    fx = "".join(f"<div><b>{html.escape(a)}</b><span>{html.escape(b)}</span></div>" for a, b in facts)
    secs = "".join(f"<section><h2>{t}</h2><p>{c}</p>{h}</section>" for t, c, h in charts)
    body = (f'<a href="../index.html">All races</a><h1>{html.escape(meta["name"])}</h1>'
            f'<p>{meta["year"]} season, round {meta["round"]}</p><div class="facts">{fx}</div>{secs}'
            '<footer>Data: OpenF1 (unofficial, from official F1 timing). Fuel-corrected at 0.03 s per lap; green-flag laps only.</footer>')
    return shell(meta["name"], body)


def build_index(manifest):
    items = "".join(
        f'<li><a href="races/{m["slug"]}.html">{html.escape(m["name"])} {m["year"]}</a>'
        f'<span>Round {m["round"]} · Winner: {html.escape(m["winner"])}</span></li>'
        for m in sorted(manifest, key=lambda m: (m["year"], m["round"]), reverse=True))
    body = f"<h1>F1 race reports</h1><p>Race pace, tyre wear and strategy. Updated automatically after every Grand Prix.</p><ul>{items}</ul>"
    return shell("F1 race reports", body)


# ---------- main ----------
def main():
    RACES.mkdir(parents=True, exist_ok=True)
    if len(sys.argv) == 3:
        year, rnd = int(sys.argv[1]), int(sys.argv[2])
        cal = calendar(year)
        row = cal[cal["round"] == rnd].iloc[0]
    else:
        year, row = latest_race()
        rnd = int(row["round"])
    slug = f"{year}-{rnd:02d}"
    manifest = json.loads(MANIFEST.read_text()) if MANIFEST.exists() else []
    if len(sys.argv) < 3 and any(m["slug"] == slug for m in manifest):
        print(f"{slug} already published, nothing to do.")
        return

    sk, mk = int(row["session_key"]), int(row["meeting_key"])
    meeting = get("meetings", meeting_key=mk)
    name = meeting.iloc[0]["meeting_name"] if not meeting.empty else f"{row['country_name']} Grand Prix"
    drivers, laps, stints = get("drivers", session_key=sk), get("laps", session_key=sk), get("stints", session_key=sk)
    result, grid, rc = get("session_result", session_key=sk), get("starting_grid", session_key=sk), get("race_control", session_key=sk)
    if laps.empty or stints.empty or result.empty:
        raise SystemExit("OpenF1 does not have complete data for this race yet. Try again later.")

    names = dict(zip(drivers["driver_number"], drivers["name_acronym"]))
    full = dict(zip(drivers["driver_number"], drivers["full_name"]))
    colors = {r["name_acronym"]: f"#{r['team_colour']}" if isinstance(r.get("team_colour"), str) else "#8791A3"
              for _, r in drivers.iterrows()}
    laps["Driver"] = laps["driver_number"].map(names).fillna(laps["driver_number"].astype(str))
    stints["Driver"] = stints["driver_number"].map(names).fillna(stints["driver_number"].astype(str))
    df = clean_laps(laps, stints, rc)
    df["Driver"] = df["driver_number"].map(names).fillna(df["driver_number"].astype(str))
    deg = stint_degradation(df)

    top = result.sort_values("position").iloc[0]
    winner = full.get(top["driver_number"], str(top["driver_number"]))
    med = df.groupby("Driver")["tc"].median().sort_values()
    facts = [(winner, "Race winner"),
             (f"{med.index[0]}, {med.iloc[1] - med.iloc[0]:.2f}s clear" if len(med) > 1 else med.index[0],
              "Fastest race pace (median clean lap)")]
    if not deg.empty:
        worst = deg.groupby("compound")["slope"].mean().idxmax()
        facts.append((f"{worst.title()} {deg[deg.compound == worst]['slope'].mean():+.3f} s/lap", "Highest average tyre wear"))

    charts = [
        ("Race pace", "Spread of each driver's clean laps, fastest median first. A tall box means inconsistent pace.", chart_pace(df, colors)),
        ("Tyre degradation", "Each dot is one stint of 8+ laps. Higher means the tyre fell off faster.", chart_deg(deg) if not deg.empty else None),
        ("Strategy", "Every stint by compound, in finishing order.", chart_strategy(stints, result, names)),
        ("Positions gained", "Grid position minus finishing position, classified finishers only.", chart_positions(result, grid, names) if not grid.empty else None),
    ]
    charts = [(t, c, f.to_html(full_html=False, include_plotlyjs=False, config={"displayModeBar": False}))
              for t, c, f in charts if f is not None]

    meta = {"year": year, "round": rnd, "name": name, "slug": slug, "winner": winner}
    (RACES / f"{slug}.html").write_text(build_race_page(meta, facts, charts), encoding="utf-8")
    manifest = [m for m in manifest if m["slug"] != slug] + [meta]
    MANIFEST.write_text(json.dumps(manifest, indent=2))
    (DOCS / "index.html").write_text(build_index(manifest), encoding="utf-8")
    print(f"Built {slug}: {name}")


if __name__ == "__main__":
    main()
