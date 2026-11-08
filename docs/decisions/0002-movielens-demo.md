# MovieLens film demo and pipeline decisions

Date: 2026-11-08

## Confirmed scope

- The user chose the MovieLens path for research data. MovieLens-100K is used
  for development and MovieLens-1M for main experiments, as the runner guide
  specifies.
- The local site is a presentation layer for browsing and playback. It does
  not run preprocessing, graph construction, attacks, training, unlearning,
  or evaluation in the browser or web server.
- Research screens read only sanitized saved outputs. No checkpoints, vault,
  raw manifests, or hidden attacker identities enter the app.
- The retail archive remains storefront-only and is not mapped to MovieLens
  titles or used as a recommendation dataset.
- MovieLens provides titles and genres but no poster images. The demo will use
  original title cards/genre treatments unless separately licensed artwork is
  supplied.
- The film browsing interface will use familiar film-diary patterns with
  original name, styling, and assets. It will not use Letterboxd's logo or
  copied proprietary artwork.
- A targeted item becoming the most popular or the top recommendation is an
  empirical outcome. The UI will show it only if a valid saved run produces
  that result; the pipeline will not force rankings for presentation.

## Defaults and unresolved spec choices

- The CLI requires an explicit world seed. No seed has been preregistered by
  this decision record.
- On 2026-11-08, the user explicitly adopted the Part 3 proposed defaults for
  new runs: choose one fixed epoch count on MovieLens-100K development worlds;
  use the revised validation protocol; freeze the exposure cohort at the
  polished poisoned state; and use the staged path-influence estimate matching
  the staged operator. Implementations and manifests must identify these as
  adopted Part 3 choices. Earlier exploratory diagnostics are not
  retroactively reclassified.
- Open choices still requiring project-level preregistration include the
  development-selected epoch count, operating point, seeds, and validation
  thresholds for V2/V4. The guide requires G1/G2 thresholds and validity rules
  to be frozen before main runs; do not select them after seeing main outcomes.
- The guide's initial BPR, attack, influence, and unlearning settings are
  implementation assumptions; they must be recorded in the run config and
  surfaced in the manifest before they are used for research claims.
- The current UI has no valid saved research run to display. It must use an
  empty state until completed outputs pass G1, G2, required leakage and
  validation checks, and manifest checks.

## Dataset handling

The project pins the official MovieLens ZIP SHA-256 values in
`configs/datasets.json`. The local archives live in ignored
`results/datasets/`. MovieLens README terms allow research use with
acknowledgement and prohibit redistribution. Never commit the archives or
their unmodified contents.
