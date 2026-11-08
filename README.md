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
uv venv --python 3.12 --clear .venv
$py = (Resolve-Path .\.venv\Scripts\python.exe).Path
uv pip install --python $py -r requirements.txt
$env:PYTHONPATH = 'src'
& $py -c "import numpy, scipy, sklearn, torch; print('Dependencies ready')"
```

Use `$py` explicitly for the commands below. This ensures PowerShell runs the
project environment even when another Python installation or a Windows app
execution alias appears earlier on `PATH`.

The two checksum-pinned MovieLens archives belong under the ignored
`results/datasets/` folder. Nothing in the pipeline accepts the storefront
archive as research data.

1. Choose the fixed epoch count once on five MovieLens-100K development worlds:

```powershell
& $py -B -m iahu.runner select-epochs `
  --archive results/datasets/ml-100k.zip `
  --parent-seed 20261108 `
  --output results/preregistration/e-star-ml-100k-mask-corrected.json
```

2. Select the UH/BPR profile on five fresh MovieLens-100K development worlds.
The approved expanded grid has 108 settings across dimension `{32,64,128}`,
layers `{1,2,3}`, learning rate `{0.0003,0.001,0.003,0.01}`, and regularization
`{0.00001,0.0001,0.001}`; batch size stays fixed:

```powershell
$epochs = (Get-Content results/preregistration/e-star-ml-100k-mask-corrected.json | ConvertFrom-Json).selected_epochs
& $py -B -m iahu.runner select-uh-config `
  --archive results/datasets/ml-100k.zip `
  --epochs $epochs `
  --parent-seed 20261112 `
  --output results/preregistration/uh-profile-v2-ml-100k-mask-corrected.json
```

The command writes a progress file alongside the profile while the 540
development fits run. Select the profile nearest a tuned LightGCN reference on
those same five development worlds:

```powershell
& $py -B -m iahu.runner calibrate-uh-parity `
  --archive results/datasets/ml-100k.zip `
  --input-profile results/preregistration/uh-profile-v2-ml-100k-mask-corrected.json `
  --output results/preregistration/uh-profile-parity-ml-100k-mask-corrected.json
```

Then rerun G1 on the same five paired MovieLens-1M worlds. Do not reuse an
older LightGCN report after an evaluation-protocol change; refit its full
12-setting comparison against the corrected candidate mask:

```powershell
& $py -B -m iahu.runner g1 `
  --archive results/datasets/ml-1m.zip `
  --epochs $epochs `
  --parent-seed 20261109 `
  --uh-profile results/preregistration/uh-profile-parity-ml-100k-mask-corrected.json `
  --output results/gates/g1-ml-1m-mask-corrected.json
```

While G1 runs, its latest candidate and completed-seed counts are written to
`results/gates/g1-ml-1m-mask-corrected.json.progress.json`. The progress file
is removed after the final report is saved.

The G1 command returns exit code 3 when its gate fails, while preserving the
report. The prior E*, profile, and G1 scores used an incomplete ranking mask:
the visible 5% validation rows were withheld from training but not excluded
from recommendation candidates. They are superseded and must not be cited.
The corrected protocol excludes all observed training/validation items. Its
current G1 report has a 11.86% relative NDCG@20 gap (UH `0.1178370`, LightGCN
`0.1336875`), so it fails the unchanged 5% threshold. **Do not run step 4**;
the profile is not G1-eligible and no valid main run or research demo bundle
exists. The main-run gate also checks the exact profile hash and settings.

3. Run the hard convex influence validation:

```powershell
& $py -B -m iahu.runner v1 --output results/gates/v1-convex.json
```

4. Run one main-candidate world with the guide's centre identification setting
and ATK-3 at the 3% attack size:

```powershell
& $py -B -m iahu.runner run `
  --archive results/datasets/ml-1m.zip `
  --dataset ml-1m `
  --parent-seed 20261110 `
  --attack atk3_popularity `
  --alpha 0.03 `
  --construction user_history `
  --uh-profile results/preregistration/uh-profile-parity-ml-100k-mask-corrected.json `
  --g1-report results/gates/g1-ml-1m-mask-corrected.json
```

Every attempted world writes a manifest under `results/runs/`. Failed gates
remain recorded as invalid. For debugging a failed G1/G2 world, add
`--exploratory`; exploratory runs can never create a demo export.

5. Export only an eligible completed run, then start the playback app:

```powershell
$runExport = Get-ChildItem results/completed-run-exports -Filter '*.json' |
  Sort-Object LastWriteTime -Descending | Select-Object -First 1
& $py apps/demo/export_demo_bundle.py --input $runExport.FullName
& $py apps/demo/server.py --port 8000
```

Open `http://127.0.0.1:8000`. If no run passes the gates, the app keeps its
explicit empty state. The demo bundle excludes manifests, user identifiers,
hidden attacker identities, per-user evaluator labels, and checkpoints.

## Local storefront data

The optional cart/catalog screen uses a separate sanitized retail catalog. It
is not part of the MovieLens research pipeline:

```powershell
& $py apps/demo/shop_catalog.py --archive 'C:\path\to\archive.zip'
```

## Checks

```powershell
$env:PYTHONPATH = 'src'
& $py -B -m unittest discover -s tests -v
node --check apps/demo/app.js
git diff --check
```

## Current research status

The CLI now selects development epochs and a UH profile, runs G1/G2 controls, records
invalid worlds, performs visible-only staged removal/reweighting with mass and
utility rollback guards, and exports only runs marked valid. The Part 3 fixed
epoch, revised validation, frozen exposure cohort, and staged path-influence
proposals were adopted for new runs. See
[`docs/decisions/0002-movielens-demo.md`](docs/decisions/0002-movielens-demo.md)
and [`docs/decisions/0003-runner-defaults.md`](docs/decisions/0003-runner-defaults.md)
for defaults and remaining validation/preregistration limits. The runner is a
single-world execution path, not the guide's full 300-victim / 552-plan grid.
