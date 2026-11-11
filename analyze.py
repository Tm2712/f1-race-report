"""Automated F1 post-race report.

Usage:
    python analyze.py              # latest completed race
    python analyze.py 2025 7       # a specific season and round
Outputs static pages into docs/ (served by GitHub Pages).
"""
import html
import json
import sys
from pathlib import Path

import fastf1
import numpy as np
import pandas as pd
import plotly.graph_objects as go

ROOT = Path(__file__).parent
DOCS, RACES, CACHE = ROOT / "docs", ROOT / "docs" / "races", ROOT / ".cache"
MANIFEST = DOCS / "races.json"
COMPOUNDS = {"SOFT": "#E10600", "MEDIUM": "#F2B800", "HARD": "#8791A3",
             "INTERMEDIATE": "#2E9E4F", "WET": "#2F6FDE"}
FUEL_PER_LAP = 0.03  # seconds of lap time gained per lap of fuel burned (rough)
MIN_STINT_LAPS = 8
PLOTLY_JS = "https://cdn.plot.ly/plotly-2.35.2.min.js"


# ---------- choosing the race ----------
def latest_race():
    now = pd.Timestamp.now(tz="UTC").tz_localize(None)
    for year in (now.year, now.year - 1):
        sched = fastf1.get_event_schedule(year, include_testing=False)
        done = sched[sched["Session5DateUtc"] + pd.Timedelta(hours=4) < now]
        if not done.empty:
            return year, int(done.iloc[-1]["RoundNumber"])
    raise SystemExit("No completed race found.")


# ---------- analysis ----------
def team_colors(results):
    out = {}
    for _, r in results.iterrows():
        c = str(r.get("TeamColor") or "")
        out[r["Abbreviation"]] = f"#{c}" if len(c) == 6 else "#8791A3"
    return out


def clean_laps(laps):
    """Green-flag racing laps only: no lap 1, pit in/out, safety cars, or slow outliers."""
    df = pd.DataFrame(laps.pick_wo_box())
    df = df[(df["TrackStatus"] == "1") & (df["LapNumber"] > 1)].dropna(subset=["LapTime", "TyreLife"])
    df["t"] = df["LapTime"].dt.total_seconds()
    df = df[df["t"] <= 1.07 * df["t"].quantile(0.10)]
    total = laps["LapNumber"].max()
    df["tc"] = df["t"] - FUEL_PER_LAP * (total - df["LapNumber"])  # fuel-corrected
    return df


def stint_degradation(df):
    rows = []
    for (drv, _), g in df.groupby(["Driver", "Stint"]):
        comp = g["Compound"].iloc[0]
        if len(g) < MIN_STINT_LAPS or comp not in COMPOUNDS:
            continue
        slope = np.polyfit(g["TyreLife"], g["tc"], 1)[0]
        rows.append({"driver": drv, "compound": comp, "slope": slope, "laps": len(g)})
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


def chart_strategy(laps, results):
    order = list(results.sort_values("Position")["Abbreviation"])
    st = laps.groupby(["Driver", "Stint", "Compound"]).agg(start=("LapNumber", "min"), end=("LapNumber", "max")).reset_index()
    fig = go.Figure()
    for comp in [c for c in COMPOUNDS if c in set(st["Compound"])]:
        d = st[st.Compound == comp]
        fig.add_trace(go.Bar(y=d["Driver"], x=d["end"] - d["start"] + 1, base=d["start"] - 1, orientation="h",
                             name=comp.title(), marker=dict(color=COMPOUNDS[comp], line=dict(color="#fff", width=2)),
                             hovertemplate="%{y}: laps %{base}–%{customdata}<extra></extra>", customdata=d["end"]))
    fig.update_yaxes(categoryorder="array", categoryarray=order, autorange="reversed")
    fig.update_xaxes(title="Lap")
    fig.update_layout(barmode="overlay")
    return style(fig, 620)


def chart_positions(results):
    r = results.sort_values("Position").copy()
    grid = r["GridPosition"].replace(0, len(r))  # pit-lane starts
    r["gain"] = grid - r["Position"]
    fig = go.Figure(go.Bar(x=r["Abbreviation"], y=r["gain"],
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
            '<footer>Data: FastF1 / official F1 timing. Fuel-corrected at 0.03 s per lap; green-flag laps only.</footer>')
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
    CACHE.mkdir(exist_ok=True)
    RACES.mkdir(parents=True, exist_ok=True)
    fastf1.Cache.enable_cache(str(CACHE))
    year, rnd = (int(sys.argv[1]), int(sys.argv[2])) if len(sys.argv) == 3 else latest_race()
    slug = f"{year}-{rnd:02d}"
    manifest = json.loads(MANIFEST.read_text()) if MANIFEST.exists() else []
    if len(sys.argv) < 3 and any(m["slug"] == slug for m in manifest):
        print(f"{slug} already published, nothing to do.")
        return

    session = fastf1.get_session(year, rnd, "R")
    session.load(laps=True, telemetry=False, weather=False, messages=False)
    laps, results = session.laps, session.results
    colors = team_colors(results)
    df = clean_laps(laps)
    deg = stint_degradation(df)

    top = results.sort_values("Position").iloc[0]
    med = df.groupby("Driver")["tc"].median().sort_values()
    facts = [(top["FullName"], "Race winner"),
             (f"{med.index[0]}, {med.iloc[1] - med.iloc[0]:.2f}s clear" if len(med) > 1 else med.index[0], "Fastest race pace (median clean lap)")]
    if not deg.empty:
        worst = deg.groupby("compound")["slope"].mean().idxmax()
        facts.append((f"{worst.title()} · {deg[deg.compound == worst]['slope'].mean():+.3f} s/lap", "Highest average tyre wear"))

    charts = [
        ("Race pace", "Spread of each driver's clean laps, fastest median first. A tall box means inconsistent pace.", chart_pace(df, colors)),
        ("Tyre degradation", "Each dot is one stint of 8+ laps. Higher means the tyre fell off faster.", chart_deg(deg) if not deg.empty else None),
        ("Strategy", "Every stint by compound, in finishing order.", chart_strategy(laps, results)),
        ("Positions gained", "Grid position minus finishing position.", chart_positions(results)),
    ]
    charts = [(t, c, f.to_html(full_html=False, include_plotlyjs=False, config={"displayModeBar": False}))
              for t, c, f in charts if f is not None]

    meta = {"year": year, "round": rnd, "name": session.event["EventName"], "slug": slug, "winner": top["FullName"]}
    (RACES / f"{slug}.html").write_text(build_race_page(meta, facts, charts), encoding="utf-8")
    manifest = [m for m in manifest if m["slug"] != slug] + [meta]
    MANIFEST.write_text(json.dumps(manifest, indent=2))
    (DOCS / "index.html").write_text(build_index(manifest), encoding="utf-8")
    print(f"Built {slug}: {meta['name']}")


if __name__ == "__main__":
    main()
