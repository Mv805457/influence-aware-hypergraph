# Local MovieLens demo

This local app has two separate parts:

- **Films** browses the approved MovieLens title/genre catalog and keeps a
  watchlist in browser memory. It does not collect or send user activity.
- **Hypergraph, Poisoning, and Results** are viewers for sanitized,
  precomputed exports. They never launch research jobs.

MovieLens contains title and genre text, not poster artwork. The cards are
generated from titles and genres; no fabricated posters are used. The app has
an original film-journal design and does not use Letterboxd branding or assets.

## Requirements

- Python 3.10 or newer. On Windows, use the official CPython distribution; an
  MSYS Python can work for the app itself, but may not support the numerical
  wheels required by research modules.
- The read-only app and catalog exporter use only the standard library. For
  numerical research code, create a Windows CPython environment and install
  `requirements.txt` as shown under Checks.
- Download MovieLens directly from GroupLens. The dataset archive stays local
  under the ignored `results/datasets/` directory and is checked against the
  repository's SHA-256 allowlist.

## Build the film catalog

From the repository root in PowerShell:

```powershell
New-Item -ItemType Directory -Force results/datasets | Out-Null
Invoke-WebRequest https://files.grouplens.org/datasets/movielens/ml-1m.zip -OutFile results/datasets/ml-1m.zip
$env:PYTHONPATH = 'src'
python -B -m iahu.movie_catalog --dataset ml-1m --archive results/datasets/ml-1m.zip --world-seed 2026
```

For the smaller development set, download `https://files.grouplens.org/datasets/movielens/ml-100k.zip` and run the same command with `--dataset ml-100k --archive results/datasets/ml-100k.zip`.

The catalog includes all approved movie titles and genres, plus popularity
counts from the clean positive training split. It excludes ratings, user IDs,
demographics, and interaction histories.

## Start the local viewer

```powershell
python apps/demo/server.py --port 8000
```

Open `http://127.0.0.1:8000`. If port 8000 is occupied, choose another local
port, such as `8765`. Stop the server with `Ctrl+C`.

The viewer has fixed GET routes for static assets and three local JSON files.
It does not accept file paths, writes, training requests, or uploaded data.
With no `results/demo-bundle.json`, the research sections show an explicit
empty state.

## Current research pipeline status

The repository has dataset preprocessing, user-history and HOUR-inspired
hypergraph builders, a BPR baseline, fake-account attacks, and ranking
evaluation modules. These do not yet make a complete, valid IAHU experiment:
the run orchestrator, identification, influence estimation, unlearning,
recovery/validation gates, complete manifest, and sanitized export from a
completed valid run are not implemented. There is no command that creates a
research result bundle yet. Do not put the exploratory diagnostic run into
the viewer or treat its failed exposure gate as a successful result.

The guide's proposed fixed-epoch and validation choices, frozen exposure
cohort, and staged path-influence estimate remain unresolved in
`docs/decisions/0002-movielens-demo.md`. No headline claim should rely on them
without team sign-off.

## Checks

Standard-library demo checks:

```powershell
python -B -m unittest tests.test_demo tests.test_movie_catalog -v
```

Full numerical and demo checks, after installing the research requirements:

```powershell
uv python install 3.12
uv venv --python 3.12 .venv
.\.venv\Scripts\Activate.ps1
uv pip install -r requirements.txt
$env:PYTHONPATH = 'src'
python -B -m unittest discover -s tests -v
node --check apps/demo/app.js
```

No data archive, generated catalog, model state, or result bundle is committed.
