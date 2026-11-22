# Additive research experiment plan

## Existing interfaces and storage

The current extension points are `SARGrid.from_sender_observation` for
sender-evidence particles, `ParticleReceiverEstimator.sample_states`,
`PairedRolloutEstimator.estimate` for matched send/no-send rollouts,
`CounterfactualOracle.evaluate` for privileged offline values,
`select_communication` and `shortlist` for baseline selection, and the
metrics/bootstrap helpers in `evaluation.metrics`. Existing runs store
`episodes.json`, `summary.json`, `config_used.json`, and a `graphs` gallery in
timestamped result folders.

New code will live in `src/voc_sar/research_experiments/`; each experiment
module will expose one runner and one experiment ID. Configs will live in
`configs/experiments/`. A dispatcher will provide one command per ID. Results
will use new `results/experiments/<ID>/<timestamp>/` directories with the same
JSON, config snapshot, and graph conventions. Report tables and figures will
go to matching new `reports/experiments/<ID>/<timestamp>/` folders.

No existing path, default, config, result, result schema, or seed list will be
modified. New configs will copy the first ten existing seeds in the same order;
an optional extended list will append seeds 10 through 29. New code that
chooses actions will consume `Observation`, candidate `Message` objects, and
public policy interfaces only. Full simulator state remains restricted to
offline evaluation/oracle code. Bootstrap intervals use 10,000 resamples.

## Experiment hooks

| ID | Additive implementation and measurements |
| --- | --- |
| E1 message-conditioned rollouts | Add a state-estimator wrapper that conditions candidate-compatible victim/hazard facts within sampled particles, then reuse paired rollouts; compare MAE, receiver/pair regret, and return with unconditioned estimates. |
| E2 role-only baseline | Add a fixed message-type/status by receiver-role compatibility table and a budget-matched episode control; compare against aware and agnostic policies. |
| E3 particle diagnostics | Sweep particle count and log particle ESS/degeneracy alongside MAE and regret; explicitly define the offline weighting diagnostic. |
| E4 noise and CRN | Add an isolated rollout evaluator with selectable common random numbers; sweep rollout count, compare against a high-count reference, and record paired-difference SE and decision runtime. |
| E5 continuation bias | Add an offline rollout continuation hook for communication-disabled, fixed-frequency, and aware continuation; compare oracle values by MAE, rank correlation, and top-1 agreement. |
| E6 robust selection/null | Add LCB and epsilon decision rules in a new selector module; construct redundant-message null contexts, then report false positives and cost sensitivity. |
| E7 cost/Pareto | Sweep the finer lambda grid, report the non-dominated return/message frontier, and add receiver-distance cost only behind a new experiment flag. |
| E8 hypothesis robustness | Sweep H1 thresholds, define delayed-value H3 contexts in config, bootstrap H2 budget-matched effects, and apply Holm correction to H1–H4. |
| E9 horizon sweep | Re-evaluate candidate policies across horizons 1, 3, 5, 10, and 15; report oracle agreement, MAE, and return. |
| E10 staleness | Pair information age with offline observed marginal value; compare Spearman correlation against the configured hazard-survival heuristic. |
| E11 environment robustness | Run low/default/high move/toggle settings and supported layout/team variants with same-seed paired aware-minus-agnostic effects. |
| E12 failures/runtime | Collect offline context-level failure traces and message metadata, preserve five examples, and time particle sampling, rollouts, and oracle evaluation separately. |

## Reports and validation

A shared additive report writer will derive CSV and LaTeX tables plus PDF/PNG
figures from each experiment's stored JSON, never from transient in-memory
results alone. New tests will check each new rule/estimator, deterministic
small-seed behavior, output isolation, and that decentralized experiment
decision paths do not read `SARGrid.positions`, `victims`, or `hazards`.

Implementation order: shared run/report helpers; E1–E4 estimator work; E5–E8
oracle/selection/statistical work; E9–E12 study runners; configs, command
dispatcher, report generation, and tests. Expensive full-seed runs will be
launched only after the corresponding small integration tests pass. Any
offline-only truth access will be labeled and kept out of action selection.
