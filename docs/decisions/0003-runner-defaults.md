# Offline runner defaults and gate status

Date: 2026-11-08

## Confirmed research scope

- The user approved preserving Parts 1–2 and using the local cart, pipeline,
  poisoning, and results screens as presentation/playback layers.
- The user adopted the Part 3 proposals for new experiments: one fixed epoch
  count selected on MovieLens-100K, the revised validation protocol, a frozen
  exposure cohort, and staged path influence. Prior exploratory runs are not
  reclassified.
- The browser app reads only `results/demo-bundle.json`. Training, poisoning,
  evaluator metrics, and unlearning remain offline.

## Defaults implemented from the runner guide

These are implementation assumptions sourced from the guide; they are stored
in run manifests and must be reviewed before external research claims:

- Five child seeds are spawned from one parent for world, attack, flagging,
  training, and unlearning.
- The post-injection visible validation split is a uniform 5% row sample.
- G1 uses a 5% relative NDCG@20 gap, under 3% relative standard deviation,
  float64 finite-difference relative error under `1e-5`, and repeated-parameter
  difference at most `1e-6`.
- G2 uses absolute target-exposure gain `0.02`, at least `5x` the reference
  exposure when the reference is nonzero, and at least `0.95` NDCG retention.
- The run CLI defaults to MovieLens-1M, ATK-3, 3% injection, UH, and the centre
  identification setting: 50% malicious recall, 10% false-positive share,
  obvious-first identification, uniform false positives, and hard confidence.
- The staged unlearner defaults to removal, extension ratio 1, mass cap 0.01,
  utility tolerance 0.02, four stages, 5 stage epochs, and up to 50 recovery
  epochs with patience 5. It uses the guide's BPR defaults and a one-negative
  frozen objective pending profiling.
- Influence reports the trapezoidal change along committed stages, plus the
  final frozen-cohort exposure surrogate with and without selected extensions.
  The latter pair is a diagnostic for shrinking-cohort artifacts; it is not
  the separately measured target exposure@20.
- The parent seed `20261108` is a reproducible engineering default, not a
  claim that the project's main seed list has been signed off.
- The current development selection uses parent seed `20261108`; the G1 command
  uses parent seed `20261109`; the documented example main world uses parent
  seed `20261110` so the development, gate, and example main streams differ.
- The initial guide UH/BPR values are starting settings. The runner now has an
  explicit 12-setting dimension/layer/learning-rate selection on five
  MovieLens-100K development worlds at the frozen E*, with batch size and
  regularization fixed. Main worlds and G1 must carry the resulting profile;
  they reject profiles trained on a different dataset, fixed epoch count, or
  setting grid. The current MovieLens-1M LightGCN comparison report can be
  reused only when its dataset hash, E*, parent seed, five world seeds, and
  12-setting budget match exactly.

## Executed development validation

- The local checksum-pinned MovieLens-100K archive was used on five seeded
  development worlds, considering epochs 1–20. **E* = 3** by mean validation
  NDCG@20 (`0.1574728679`); the complete score table and independent child
  seeds are saved in ignored `results/preregistration/e-star-ml-100k.json`.
- UH/BPR selection used the documented 12-setting grid on five separate
  MovieLens-100K development worlds at E*=3. It selected dimension 64, one
  propagation layer, learning rate 0.003, regularization 0.0001, and batch
  size 4096 (mean validation NDCG@20 `0.1772994602`). The full table is saved
  in ignored `results/preregistration/uh-profile-ml-100k.json`.
- The profiled G1 was rerun on the original five paired MovieLens-1M seeds,
  reusing the exact prior LightGCN comparison results. Mean UH NDCG@20 was
  `0.1212943491` versus `0.1291315584` for LightGCN, a 6.07% relative gap
  against the unchanged 5% limit. Variance (2.26%), finite-difference gradient
  (1.26e-7), and same-seed reproducibility checks passed; G1 remains failed.
  The full report is ignored at `results/gates/g1-ml-1m-profiled.json`.
- V1 uses a convex SPD quadratic toy and compares the production CG/implicit
  derivative path with exact solve-based retraining. The local run passed with
  maximum source relative error `2.06e-10`; report saved at
  `results/gates/v1-convex.json`.

## Still open or not yet run

- Both the original G1 (13.06% gap) and the development-profiled G1 (6.07%
  gap) fail the relative NDCG gate. Do not run a main-candidate attack against
  a failed G1. A new profiling decision would require development-only tuning
  with its full candidate set recorded; do not weaken the G1 threshold. G2 is
  an empirical attack strength gate and is not changed to make a world pass.
- V2 (200 leave-one-source-out cases plus attackers), V3 group-removal
  validation, and V4 placebo distinguishability are not implemented as a full
  validation campaign yet. Their thresholds were not stated in the guide; no
  headline claim about influence ranking is permitted until the team freezes
  those thresholds and observes results.
- The CLI executes one world at a time. It does not yet orchestrate the full
  300 victim trainings, 552 core-plan settings, tuning tiers, bootstrap tables,
  resume/replay guarantees, or all required validation strata.
- The attack, model, unlearning, and influence hyperparameters other than the
  explicit guide defaults need a Phase 1 profile and a signed preregistration
  before they are used in a research comparison.
- OQ-1, sparse-dataset selection, the exact main seed list, operating-point
  selection, V2/V4 thresholds, the H3 margin, and whether OQ-2 degree matching
  is usable remain project decisions.

## Export policy

The runner writes a sanitized source export only when G1, G2, V1, the static
unlearning import-boundary check, and influence-reliability checks all pass.
Invalid and exploratory worlds retain aggregate reports/manifests and are
never copied to the demo export directory. The exporter then applies the
existing strict field whitelist.
