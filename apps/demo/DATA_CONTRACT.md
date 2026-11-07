# Read-only demo data contract

`results/demo-bundle.json` is the only data file read by the app. The research
pipeline and evaluator remain offline. The demo does not load checkpoints, read
run manifests, or compute metrics.

## Bundle format

The root object is:

```json
{
  "schema_version": 1,
  "scenarios": []
}
```

Each scenario contains only:

- A sanitized display `label` and the fixed `tier` value `deployable`. Source
  run/manifest identifiers are discarded.
- `setting`: `dataset`, `hypergraph_construction`, `attack`, the five independent
  seeds (`world`, `attack`, `flagging`, `training`, `unlearning`), and the
  identification setting (`rate`, `false_positive_share`, `strategy`). These are
  displayed as offline experiment settings, never as runtime inputs.
- `target_item_label` and one `example` with the exporter-assigned pseudonym
  `Example user 01`, `top_k`,
  the pre-unlearning `before` list, and `target_rank_before`.
- `methods`: a sanitized method `label` (source method IDs are discarded), the
  `after` list and `target_rank_after`,
  optional aggregate metrics, and `elapsed_seconds`.

Top-K list entries contain only a display `label` and `is_target`. Rank is one based.
A `null` target rank means the full rank was not exported and the target is absent
from that saved Top-K list. A rank greater than K means the target was outside the
saved list. The app does not infer a full rank from an absent item.

Aggregate metric keys are a fixed whitelist in `contract.py` (RAR, hidden-source
recall, extension precision, benign weight loss, utility retention, control-item
drift, NDCG@20, Recall@20, Top-K Jaccard, cost ratio, and membership AUC gap).
Only metrics present in a completed result are shown. They are labelled as
aggregate evaluator metrics and are never treated as deployable inputs. Elapsed
time is shown in seconds.

The structure intentionally excludes user IDs, per-user evaluator labels, hidden
attacker identities, vault contents, raw manifests, file paths, model checkpoints,
per-item recommendation scores, and training data. Aggregate evaluation metrics
are retained only from the explicit whitelist above. The single example must
already be selected offline from the benign evaluation cohort. The exporter
replaces any source user label with the fixed pseudonym. The bundle contains no
per-user class label or identity; the page frames the pseudonymous sample as a
curated evaluation example, not as a runtime input.

Scenarios are tier-separated: this demo accepts only `deployable` results and
rejects oracle-tier results. It does not combine rows from different tiers.
The whitelist protects fields and structure; it cannot recognize sensitive text
mistakenly placed inside an approved display-label field. The offline producer
must source those labels from reviewed configuration and item-display fields,
not from raw manifests or user records.

## Export input

`apps/demo/export_demo_bundle.py` accepts a UTF-8 JSON file at
`results/completed-run-exports/completed-run-export.json` with
`schema_version: "completed-run-export-v1"` and a `scenarios` array shaped like
the bundle format. An alternate `--input` must still resolve inside
`results/completed-run-exports/`; the exporter cannot read from the vault or other
paths. The output path is fixed to `results/demo-bundle.json`. The exporter copies
only whitelisted public fields, validates them, and drops all extra keys at every level.
It does not derive or fabricate results. The empty source repository has no
experiment pipeline to produce this input yet; the offline runner must write it
from completed runs before the demo can show results.

## Empty and invalid data

An empty `scenarios` array is valid and displays an explicit no-results message.
If the generated bundle is missing, the server returns that same explicit empty
state. Malformed, unsupported, or otherwise invalid data is reported in the page
with a readable explanation. No example metrics or recommendations are bundled.
