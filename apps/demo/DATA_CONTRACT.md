# MovieLens demo data contract and boundary

The browser and local HTTP server are presentation only. The server serves a
small fixed set of static assets and reads only these fixed paths under the
ignored `results/` directory:

- `movie-catalog.json`: sanitized titles, genres, years, and clean-train
  popularity for the film browsing page.
- `demo-bundle.json`: a sanitized bundle exported from a completed, valid
  offline experiment.

The server exposes fixed GET routes only. It does not resolve user-supplied
paths, import `src/iahu`, open datasets, train a model, read a checkpoint, or
access a vault. The browser cannot submit events to an experiment. Its
watchlist and watched marks exist only in tab memory.

## Movie catalog

Schema version 1 has exactly these root fields:

```json
{
  "schema_version": 1,
  "dataset": "ml-1m",
  "dataset_role": "main",
  "dataset_sha256": "<allowlisted archive digest>",
  "popularity_source": "positive MovieLens training interactions before attack",
  "world_seed": 2026,
  "movies": [
    {"id": 1, "title": "Toy Story (1995)", "year": 1995,
     "genres": ["Animation", "Children's", "Comedy"], "popularity": 423}
  ]
}
```

Movie objects contain only the MovieLens item ID, public catalog metadata, and
clean-train positive-interaction popularity. Ratings, user IDs, demographics,
timestamps, and user histories are excluded. The item ID is used locally to
join a recommendation to its title; it is not an account identifier. MovieLens
has no poster images, so the page uses original text-led cards rather than
fabricated artwork.

The catalog is generated offline after SHA-256 verification against
`configs/datasets.json`. MovieLens archives remain local and are not committed
or redistributed.

## Experiment bundle

Schema version 2 is deliberately empty until a valid completed-run exporter
exists and a run passes the required gates:

```json
{"schema_version": 2, "scenarios": []}
```

Every scenario contains a reviewed display label, `tier: "deployable"`, the
dataset, hypergraph construction, attack, five seeds, and identification
setting; a target display name; one pseudonymous benign evaluation example
with before Top-K and target rank; fixed-step aggregate traces; and one or more
methods with their saved after Top-K, target rank, allowed aggregate metrics,
and elapsed seconds. Top-K entries contain only an item display label and a
boolean target marker. Ranks are one-based full-candidate ranks; `null` means
the rank was unavailable, while a rank above K means the target is outside the
displayed list.

Metric names are whitelisted in `contract.py`, including residual attack
ratio, hidden-source recall, precision, benign weight loss, utility retention,
control-item drift, NDCG@20, Recall@20, Top-K Jaccard, cost ratio, and
membership AUC gap. The UI labels metrics and elapsed time and does no
evaluation itself.

Pipeline and poisoning traces are playback events only. Each event has a
fixed step name, whitelisted aggregate counts, and elapsed seconds. No IDs,
per-user evaluator labels, per-item scores, hidden attacker identities, vault
contents, raw manifests/paths, or checkpoints may appear in the bundle. The
offline exporter assigns `Example user 01`; this is a curated evaluation
example, not a deployable user input.

The current repository does **not** yet have a completed-run producer for
this schema. The exploratory MovieLens diagnostic did not pass its target
exposure gate and is intentionally not exported. Keep the result view empty
until a valid run is produced; do not copy test fixtures or diagnostic values
into `results/demo-bundle.json`.

## Completed-run source and export

The sanitizer expects a JSON source with
`schema_version: "completed-run-export-v1"` beneath
`results/completed-run-exports/`. It copies only the approved fields and
writes `results/demo-bundle.json`. The current exporter is a sanitizer, not a
run producer, and its test fixtures are not experiment evidence. The valid
source format must only be populated from saved offline outputs after all
validity gates and manifest checks pass.

## Missing or malformed files

Missing catalogs and bundles return empty structures; the page reports an
empty state. Invalid JSON or contract violations return a readable HTTP 422
response. Only the fixed routes `/api/movies`, `/api/catalog` (legacy retail
catalog), and `/api/bundle` are dispatched; other paths are rejected. The
legacy retail catalog is separate from MovieLens and is not used as research
input.
