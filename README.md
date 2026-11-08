# Influence-Aware Hypergraph Unlearning

Research code and a local, read-only MovieLens film-journal demo for the
capstone project. The browser does not run preprocessing, poisoning, training,
unlearning, or evaluation.

## Run the local film demo

See [apps/demo/README.md](apps/demo/README.md) for the full PowerShell steps.
In short, download the approved MovieLens-1M archive into the ignored
`results/datasets/` directory, generate `results/movie-catalog.json`, then run:

```powershell
python apps/demo/server.py --port 8000
```

Open `http://127.0.0.1:8000`. Without a valid saved experiment bundle, the
research playback sections remain empty by design.

## Research status

MovieLens ingestion and splitting, two sparse hypergraph constructors, a
linear BPR model, controlled fake-account attacks, and ranking evaluation are
implemented. The end-to-end experiment runner and several IAHU stages/gates
are still missing; no valid completed-run bundle is currently available. See
[`docs/decisions/0001-gap-list.md`](docs/decisions/0001-gap-list.md) for the
current inventory and [`apps/demo/DATA_CONTRACT.md`](apps/demo/DATA_CONTRACT.md)
for the demo boundary.

## Tests

```powershell
uv python install 3.12
uv venv --python 3.12 .venv
.\.venv\Scripts\Activate.ps1
uv pip install -r requirements.txt
$env:PYTHONPATH = 'src'
python -B -m unittest discover -s tests -v
node --check apps/demo/app.js
```

Do not commit dataset archives, generated results, or model state.
