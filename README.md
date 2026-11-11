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
