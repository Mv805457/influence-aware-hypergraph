# Local precomputed demo

This optional interface is isolated under `apps/demo/`. It shows only results
already exported from completed offline experiments. It does not run training,
poisoning, unlearning, or evaluation, and does not load a model checkpoint or
import the research package.

## Start

From the repository root:

```powershell
python apps/demo/server.py
```

Open <http://127.0.0.1:8000>. Stop the server with `Ctrl+C`. The server binds only
to loopback and serves the interface assets plus `results/demo-bundle.json`.

## Export completed results

The repository currently has no offline pipeline or completed run exports. When
an offline evaluator produces a file matching
[`DATA_CONTRACT.md`](DATA_CONTRACT.md), export it with:

```powershell
python apps/demo/export_demo_bundle.py --input results/completed-run-exports/completed-run-export.json
```

The input must be inside `results/completed-run-exports/`; the output is fixed to
`results/demo-bundle.json`. The exporter preserves only the
contract fields, accepts deployable-tier rows only, and does not calculate or
invent metrics. It expects one pre-curated pseudonymous example per scenario.
Oracle results and per-user evaluator labels are not accepted as display fields.

An empty `results/demo-bundle.json` is valid. The app will show an explicit empty
state until the offline pipeline supplies completed runs. The app is useful for
checking the presentation shell before results exist, but it cannot demonstrate
the research effect without a valid exported run.
