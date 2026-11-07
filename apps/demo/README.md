# Local four-screen demo

The optional interface in `apps/demo/` has four screens: a local product/cart
mockup, saved hypergraph-pipeline playback, saved poisoning/unlearning playback,
and a precomputed Top-K results viewer. The cart is browser-memory-only. The
three research screens read sanitized artifacts from `results/`; they never run
data processing, graph construction, poisoning, training, unlearning, or
evaluation. The server binds to loopback and exposes only fixed GET routes.

This preserves the current research specification. The supplied architecture
diagram's Amazon Video Games dataset and proposed model/attack/unlearning blocks
are not activated by this demo. The supplied `archive.zip` is used only to fill
the storefront catalog; its data does not generate research results.

## 1. Create the local storefront catalog

From the repository root, import the attached archive (adjust the path for your
machine):

```powershell
python apps/demo/shop_catalog.py --archive "C:\Users\madha\Downloads\archive.zip"
```

The importer reads `amazon.csv` directly from the ZIP without extracting it. It
writes `results/shop-catalog.json`, which is gitignored. The catalog whitelist
contains generated local product IDs, product names, category, prices, rating,
rating count, and discount percentage. Source product IDs are used transiently
for deduplication and discarded. Reviews, user identifiers/names, review text,
images, links, and descriptions are excluded. Cart changes are not saved.

The supplied archive is a mixed retail catalog with no Video Games category and
does not meet the experiment's stated 10-core requirement. It must not be used
as evidence for the research experiment.

## 2. Supply completed offline experiment exports

The repository currently contains no offline runner, completed experiment run,
or trace producer. There are no valid research results to display. Do not create
sample metrics or recommendations to populate the viewer.

Once the offline evaluator has completed a run, it must write a reviewed JSON
file matching [`DATA_CONTRACT.md`](DATA_CONTRACT.md) to:

```text
results/completed-run-exports/completed-run-export.json
```

Then export it with:

```powershell
python apps/demo/export_demo_bundle.py --input results/completed-run-exports/completed-run-export.json
```

The exporter writes `results/demo-bundle.json` (schema version 2). It accepts
deployable-tier results only and copies fields from explicit whitelists. Trace
events are optional. If the offline pipeline has not recorded a trace, its
screen explains that the trace is unavailable. The exporter does not calculate
or invent values. Never include a vault path, hidden attacker identity, raw
manifest, checkpoint path, or evaluator-only per-user label in the source file.

## 3. Run the demo

```powershell
python apps/demo/server.py --port 8000
```

Open <http://127.0.0.1:8000>. The four screens are:

1. **Shop:** searchable product cards, category filter, quantity controls,
   temporary cart, subtotal, and a demo-only checkout notice. It creates no
   order and clears when the page reloads.
2. **Hypergraph:** playback controls for exported `pipeline_trace` events.
3. **Poisoning:** playback controls for exported `poisoning_trace` events.
4. **Results:** same pseudonymous example user and target before/after Top-K,
   target ranks, saved setting, aggregate metrics, and method elapsed time.

Playback advances only through stored event records. There is no live job UI or
connection to the experiment package, model checkpoints, or vault. An empty
catalog, missing bundle, or missing trace has a readable empty state; malformed
or unsupported files are reported as errors.

## Checks

```powershell
python -B -m unittest discover -s tests -v
node --check apps/demo/app.js
```

Generated catalogs, completed-run exports, and demo bundles live under ignored
`results/` and should not be committed.
