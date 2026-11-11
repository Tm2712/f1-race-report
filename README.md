# F1 race reports

Builds a post-race analysis page after every Grand Prix: race pace, tyre degradation, strategy, positions gained.
Runs on GitHub Actions and publishes to GitHub Pages. Free, no server.

## Deploy (about 10 minutes)
1. Create a new **public** GitHub repo and upload everything in this folder (keep the `.github` folder).
2. Repo **Settings > Pages**: Source = *Deploy from a branch*, Branch = `main`, Folder = `/docs`.
3. Repo **Settings > Actions > General > Workflow permissions**: choose *Read and write permissions*.
4. Go to the **Actions** tab, open *Build race report*, click **Run workflow** (leave inputs blank).
5. When it finishes, your site is at `https://<your-username>.github.io/<repo-name>/`.

From then on it runs by itself every Sunday evening and Monday morning, and skips races it has already published.

## Run locally
    pip install -r requirements.txt
    python analyze.py            # latest race
    python analyze.py 2025 7     # a specific race
Open `docs/index.html` in a browser.

## Ideas for version 2
- Qualifying report and teammate head-to-head
- Pit stop time loss per team
- Season-long pace rankings built from the stored race data
- A post to Discord or X from the workflow

## Method and limitations
- Data: the free OpenF1 API (historical data, no login). It is unofficial; timestamps are approximate.
- Pace and tyre degradation use green-flag laps only (no lap 1, in/out laps, safety cars or laps over 107% of the
  fast pace) with a rough fuel correction of 0.03 s per lap.
- Undercut analysis: a stop counts when the car was within 3 s behind a car that had not pitted more recently; the
  result is judged two laps after the car ahead responds. It ignores tyre offsets, traffic and team intent.
- Pace-vs-result compares median clean-lap pace rank with finishing position and only flags gaps of 3+ places.
- `python tests/test_insights.py` runs checks on synthetic race data (needs pandas only).
