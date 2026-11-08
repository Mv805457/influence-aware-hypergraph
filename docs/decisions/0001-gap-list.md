# Offline runner inventory and current validity status

Last reviewed: 2026-11-08

## Implemented locally

- S0 data: SHA-256 allowlisted MovieLens-100K/1M ingestion, positive
  binarization, iterative 10-core filtering, seeded per-user split, and target
  candidate selection.
- S1 construction: user-history and explicitly HOUR-inspired NMF
  hypergraphs; sparse matrix-free propagation with frozen/rebuilt ownership
  modes.
- Baseline: linear graph/hypergraph propagation with BPR training and a
  LightGCN operator for comparison.
- Attack/evaluation primitives: seeded ATK-1 through ATK-5 fake accounts,
  matched placebo injection, hidden-ID relabeling in memory, full-ranking
  NDCG/Recall, target exposure, and target rank.
- S5 flagging simulator: controlled recall/noise with visible hard/soft flags
  isolated from evaluator truth, JSD anomaly scores, and attack strategies.
- Gate functions: explicit G1/G2 evaluators; unlearning plan builders for
  source removal/reweighting from visible flags plus supplied suspicion scores.
- S6 influence primitives: differentiable weight-dependent propagation,
  frozen-negative objective, L-BFGS polish, damped CG/Hessian-vector products,
  exposure influence, overlap proximity, and blended suspicion ranking.
- Single-world offline CLI: five child seed streams, MovieLens-100K E* selection,
  five-seed G1 with 12-config LightGCN tuning and a development-selected UH
  profile, paired attack/no-attack/placebo
  evaluation, G2 enforcement, stage manifests, visible-only staged unlearning,
  mass/utility rollback guards, recovery, and a static import-boundary check.
- V1 convex finite-difference influence validation in float64.
- Optimized full-ranking evaluation with deterministic item-ID tie breaking.
- Local demo: MovieLens title/genre catalog generation and fixed GET-only
  playback server with strict catalog and result-bundle contracts.

The runner writes a complete one-world manifest and writes a demo source export
only after the currently implemented hard gates pass. No valid main run has
been produced by this implementation yet.

## Blocking work before a valid research bundle

- The CLI is not the complete pre-registered grid: it does not cover all five
  targets, 300 victim trainings, 552 plan settings, tiered baseline comparisons,
  bootstrap tables, or interrupted-run resume/replay guarantees.
- V2 (200 leave-one-source-out cases plus attackers), V3 group comparisons,
  and V4 placebo indistinguishability are not implemented as a validation
  campaign. Freeze thresholds before using influence-ranking results as evidence.
- Exact main seeds, operating-point selection, H3 margin, V2/V4 thresholds,
  sparse-dataset OQ-1, OQ-2 degree matching, and research hyperparameter tuning
  still need project preregistration.
- The profiled MovieLens-1M G1 still fails its unchanged 5% NDCG-gap gate; no
  valid main run or sanitized research bundle exists. Main attack/control
  worlds are held until G1 passes.

## Observed diagnostic status (not an accepted result)

An exploratory MovieLens-1M UH/BPR + ATK-3 diagnostic was run before the new
runner. Its target mean rank improved, but target exposure in Top-20 remained
zero, so G2 did not pass. It is not a valid run and is not exported.

The new V1 convex validation passed with maximum source relative error
`2.06e-10`. E* and UH-profile selection scores are saved under ignored
`results/preregistration/`; see decision 0003. Neither result implies that G1
or G2 passes on MovieLens-1M.

## Data handling

The official MovieLens archives are checksum-verified and live only under the
ignored `results/datasets/` path. The 1,000-row retail sample supplied earlier
is catalog metadata without user-item interactions; it remains separate from
the research pipeline. MovieLens is used for development/main runs per the
runner guide, while sparse-dataset OQ-1 remains unresolved.
