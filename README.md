# Influence-Aware Hypergraph Unlearning

This repository contains the offline MovieLens experiment pipeline and an
optional local read-only demo. The demo reads only a sanitized JSON bundle. It
does not ingest research data, train models, unlearn, evaluate, or open model
checkpoints or evaluator state.

## Offline experiment commands (PowerShell)

Install the repository dependencies in a Python 3.12 environment, then set the
source path for the CLI:

```powershell
uv python install 3.12
uv venv --python 3.12 .venv
.\.venv\Scripts\Activate.ps1
uv pip install -r requirements.txt
$env:PYTHONPATH = 'src'
```

The two checksum-pinned MovieLens archives belong under the ignored
`results/datasets/` folder. Nothing in the pipeline accepts the storefront
archive as research data.

1. Choose the fixed epoch count once on five MovieLens-100K development worlds:

```powershell
python -B -m iahu.runner select-epochs `
  --archive results/datasets/ml-100k.zip `
  --parent-seed 20261108 `
  --output results/preregistration/e-star-ml-100k.json
```

2. Run the clean-model gate on five paired MovieLens-1M worlds. It tunes plain
LightGCN on the documented 12-configuration budget and writes a report even
when G1 fails:

```powershell
$epochs = (Get-Content results/preregistration/e-star-ml-100k.json | ConvertFrom-Json).selected_epochs
python -B -m iahu.runner g1 `
  --archive results/datasets/ml-1m.zip `
  --epochs $epochs `
  --parent-seed 20261109 `
  --output results/gates/g1-ml-1m.json
```

While G1 runs, its latest candidate and completed-seed counts are written to
`results/gates/g1-ml-1m.json.progress.json`. The progress file is removed after
the final report is saved.

3. Run the hard convex influence validation:

```powershell
python -B -m iahu.runner v1 --output results/gates/v1-convex.json
```

4. Run one main-candidate world with the guide's centre identification setting
and ATK-3 at the 3% attack size:

```powershell
python -B -m iahu.runner run `
  --archive results/datasets/ml-1m.zip `
  --dataset ml-1m `
  --parent-seed 20261110 `
  --attack atk3_popularity `
  --alpha 0.03 `
  --construction user_history `
  --g1-report results/gates/g1-ml-1m.json
```

Every attempted world writes a manifest under `results/runs/`. Failed gates
remain recorded as invalid. For debugging a failed G1/G2 world, add
`--exploratory`; exploratory runs can never create a demo export.

5. Export only an eligible completed run, then start the playback app:

```powershell
$runExport = Get-ChildItem results/completed-run-exports -Filter '*.json' |
  Sort-Object LastWriteTime -Descending | Select-Object -First 1
python apps/demo/export_demo_bundle.py --input $runExport.FullName
python apps/demo/server.py --port 8000
```

Open `http://127.0.0.1:8000`. If no run passes the gates, the app keeps its
explicit empty state. The demo bundle excludes manifests, user identifiers,
hidden attacker identities, per-user evaluator labels, and checkpoints.

## Local storefront data

The optional cart/catalog screen uses a separate sanitized retail catalog. It
is not part of the MovieLens research pipeline:

```powershell
python apps/demo/shop_catalog.py --archive 'C:\path\to\archive.zip'
```

## Checks

```powershell
$env:PYTHONPATH = 'src'
python -B -m unittest discover -s tests -v
node --check apps/demo/app.js
git diff --check
```

## Current research status

The CLI now selects a development epoch count, runs G1/G2 controls, records
invalid worlds, performs visible-only staged removal/reweighting with mass and
utility rollback guards, and exports only runs marked valid. The Part 3 fixed
epoch, revised validation, frozen exposure cohort, and staged path-influence
proposals were adopted for new runs. See
[`docs/decisions/0002-movielens-demo.md`](docs/decisions/0002-movielens-demo.md)
and [`docs/decisions/0003-runner-defaults.md`](docs/decisions/0003-runner-defaults.md)
for defaults and remaining validation/preregistration limits. The runner is a
single-world execution path, not the guide's full 300-victim / 552-plan grid.
