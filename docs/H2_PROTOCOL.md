# H2 Whole-Episode Matched-Budget Protocol

## Scope

This experiment tests whether receiver-aware selection produces higher task
return than receiver-agnostic selection when both methods have the same
whole-episode communication budget. It does not test unrestricted abstention
as its primary comparison. Full broadcast and never-communicate are contextual
controls.

## Locked design

- **Primary quota:** five messages per episode.
- **Secondary quota:** ten messages per episode. This is reported separately
  and cannot replace the primary result.
- **Environment:** the current default configuration at `gamma=0.9`,
  `hazard_move_probability=0.3`, and `hazard_toggle_probability=0.2`.
- **Original seed block:** 510 through 559, shared across methods and quotas.
- **Independent replication block:** 710 through 759, using the same locked
  configuration, method, quota order, eligibility rule, and analysis.
- **Policy horizon:** nonmyopic horizon 5; communication cost weight 0 for the
  aware/agnostic comparison.
- **Decision timing:** both selectors may send value-positive messages at any
  decision throughout the episode, until the quota is met. If the quota remains
  and the number of horizon steps left equals the remaining messages, the
  method sends its best available pair even if estimated net value is
  non-positive. The runner records those fallback sends separately.
- **Early termination:** if all victims are rescued before a method meets its
  quota, that episode is recorded as a quota miss. It is excluded from the
  exact-budget aware-versus-agnostic paired return analysis. Quota attainment
  counts remain reported for every attempted seed.

## Analysis

The primary H2 comparison uses paired seeds where both methods sent exactly
five messages. Report the aware-minus-agnostic discounted-return mean and its
95% paired bootstrap confidence interval. The primary return criterion is met
only if this interval is positive. For aware versus full broadcast, report the
communication reduction among same-seed episodes whose returns differ by at
most 1.0, with its 95% bootstrap interval; the target reduction is 20%.

The ten-message quota uses the same analysis as a secondary robustness check.
Report quota misses, fallback sends, and exact-pair counts. Do not pool the two
quotas or substitute the secondary result for the primary test.

## Reproduction

Run either block from the project root with the current default config. The
independent replication commands are:

```powershell
$env:PYTHONPATH='src'
python -m voc_sar.experiments.h2_budget --config configs/default.yaml --budget 5 --seed-start 710 --seed-count 50
python -m voc_sar.experiments.h2_budget --config configs/default.yaml --budget 10 --seed-start 710 --seed-count 50
```

The implementation is in `src/voc_sar/experiments/h2_budget.py`,
`src/voc_sar/experiments/runner.py`, and
`src/voc_sar/communication/selector.py`. Results are summarized in
`results/h2_full_episode_budget_consolidated.json`.
