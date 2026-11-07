# Demo data contracts and information boundary

The demo is a viewer, not part of the experiment runtime. The server reads two
fixed ignored files: `results/shop-catalog.json` for the local storefront and
`results/demo-bundle.json` for saved experiment results. It has no route for
arbitrary files. It does not import the research package, inspect run
manifests, open model checkpoints, or access the vault. Only GET is supported.

## Store catalog

`shop-catalog.json` has this shape:

```json
{
  "schema_version": 1,
  "products": [
    {
      "id": "product-00001",
      "name": "Display name",
      "category": "Category / Subcategory",
      "price": 100.0,
      "list_price": 150.0,
      "rating": 4.5,
      "rating_count": 1200,
      "discount_percent": 33.0
    }
  ]
}
```

The importer reads only product ID (transient deduplication key), product name,
category, discounted price, actual price, discount percentage, rating, and
rating count from `amazon.csv`. It creates sequential opaque local IDs and
serializes only the fields shown above. The CSV's descriptions, review data,
reviewer identities, image URLs, and product URLs are excluded. The cart exists
only in JavaScript memory and is cleared on reload. The fake checkout button
only displays a notice.

The current archive is a mixed retail sample. Its product catalog supports a
storefront visual; it is not a research dataset or an experiment output.
Its price strings use the Indian rupee symbol, so the storefront assumes INR
when displaying parsed numeric prices.

## Experiment result bundle

The bundle uses schema version 2:

```json
{
  "schema_version": 2,
  "scenarios": []
}
```

Each scenario contains:

- `label` and the fixed tier `deployable` (oracle-tier outputs are rejected).
- `setting`: dataset, hypergraph construction, attack, five independent seeds
  (`world`, `attack`, `flagging`, `training`, `unlearning`), identification
  rate, false-positive share, and strategy.
- `target_item_label` and one curated example: the exporter-assigned display
  pseudonym `Example user 01`, Top-K size, saved `before` list, and
  `target_rank_before`.
- Optional playback arrays `pipeline_trace` and `poisoning_trace`. They are
  present in bundle schema version 2; absent source traces become empty arrays.
- One or more `methods`, each with a public display label, saved `after` list,
  `target_rank_after`, available aggregate metrics, and `elapsed_seconds`.

Top-K list entries contain only `label` and `is_target`. Ranks are one-based. A
`null` target rank means the target was absent from the saved Top-K list and the
full rank was not exported. A rank greater than K means the item is outside that
list.

### Playback event schema

Each trace event contains exactly:

```json
{
  "step": "hypergraph_construction",
  "counts": {"users": 25, "hyperedges": 7},
  "elapsed_seconds": 1.25
}
```

Pipeline step codes are `dataset_preprocessing`, `interaction_matrix`,
`hypergraph_construction`, and `recommendation_baseline`. Poisoning step codes
are `attack_configuration`, `poisoning_simulation`, `identification`,
`unlearning`, and `evaluation`. Allowed aggregate counters are `users`, `items`,
`interactions`, `hyperedges`, `memberships`, `injected_interactions`,
`candidate_sources`, `flagged_sources`, `removed_hyperedges`, and
`evaluation_users`. Counts are non-negative integers. `elapsed_seconds` is a
non-negative number or `null`. The interface labels these as saved events; its
playback controls only change which saved event is highlighted.

The event labels are fixed in the interface. Unknown steps, fields, and counters
are rejected or discarded by the exporter. This demo contract does not authorize
any new experiment method. It only gives the offline run exporter a restricted
way to supply saved aggregate checkpoints.

### Metrics and identifying information

Metric keys are explicitly whitelisted in `contract.py`: residual attack ratio,
hidden-source recall, extension precision, benign weight loss, utility
retention, control-item drift, NDCG@20, Recall@20, Top-K Jaccard, cost ratio,
and membership AUC gap. Only aggregate values included in a completed result
are shown. Elapsed time is shown in seconds. No value is computed by the app.

The exporter drops run/method IDs, source example user labels, item IDs, per-user
evaluator labels, hidden attacker identities, vault contents, raw manifests,
paths, per-item scores, and checkpoints. The demo example is curated offline
from an approved benign evaluation cohort. The page describes the pseudonymous
row as an evaluation example, not a deployable runtime input. The field
whitelist cannot detect sensitive text mistakenly placed inside an approved
display label; the offline producer must use reviewed scenario labels and item
display names.

## Completed-run export input

The source JSON path must be inside `results/completed-run-exports/`, and its
root must contain `schema_version: "completed-run-export-v1"` and a `scenarios`
array with the fields above. `pipeline_trace` and `poisoning_trace` are optional
source fields. Source event objects may contain extra fields, but the exporter
copies only fixed steps, approved aggregate counters, and elapsed time before
validating the result bundle. The output path is fixed at
`results/demo-bundle.json`.

The repository currently has no offline run producer or valid completed run.
The sample archive is not suitable for the project's stated 10-core dataset
experiment. Until a valid completed run is produced, results and trace views
must remain empty; no example recommendations, trace counts, or metrics should
be added to the bundle.

## Missing and invalid files

If the catalog or result bundle is absent, the server serves an empty structure
and the page displays an explicit empty state. Missing trace arrays display
“no trace in this run.” Invalid JSON, unsupported schema versions, invalid
ranks, or unapproved bundle fields return a readable contract error. The
frontend only fetches the fixed `/api/catalog` and `/api/bundle` routes.
