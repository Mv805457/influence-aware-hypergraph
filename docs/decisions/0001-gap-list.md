# Offline runner inventory and current validity status

## Implemented locally

- S0 data: SHA-256 allowlisted MovieLens-100K/1M ingestion, positive
  binarization, iterative 10-core filtering, seeded per-user split, and target
  candidate selection.
- S1 construction: user-history and explicitly HOUR-inspired NMF
  hypergraphs; sparse matrix-free propagation with frozen/rebuilt ownership
  modes.
- Baseline: linear graph/hypergraph propagation with BPR training and a
  LightGCN operator for comparison.
- Attack/evaluation primitives: seeded ATK-1/2/3 fake accounts, matched
  placebo injection, hidden-ID relabeling in memory, full-ranking NDCG/Recall,
  target exposure, and target rank.
- Local demo: MovieLens title/genre catalog generation and fixed GET-only
  playback server with strict catalog and result-bundle contracts.

These modules are building blocks, not an end-to-end runner. The code does not
yet produce a signed, complete run manifest or the source JSON expected by the
demo bundle exporter.

## Blocking work before a valid research bundle

- Confirm or replace all unresolved Part 3 `[P3]` method proposals.
- Implement the guide-specified flag identification, influence estimator, IAHU
  unlearning, recovery, validation, and leakage checks without inventing
  formulas or silently changing methods.
- Implement seeded orchestration, run manifests, resume/validity accounting,
  comparison baselines, gate enforcement, and a sanitizer-backed exporter.
- Add numerical, leakage, validation, integration, and repeatability checks
  for those stages.

## Observed diagnostic status (not an accepted result)

An exploratory MovieLens-1M UH/BPR + ATK-3 diagnostic was run locally. Its
target mean rank improved, but target exposure in Top-20 remained zero, so G2
did not pass. It did not run the complete IAHU method or all required
acceptance gates. The diagnostic was not saved as a completed run and is not
exported to the demo bundle.

## Data handling

The official MovieLens archives are checksum-verified and live only under the
ignored `results/datasets/` path. The 1,000-row retail sample supplied earlier
is catalog metadata without user-item interactions; it remains separate from
the research pipeline. MovieLens is used for development/main runs per the
runner guide, while sparse-dataset OQ-1 remains unresolved.
