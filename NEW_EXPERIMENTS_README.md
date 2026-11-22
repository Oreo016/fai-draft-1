# Additive research studies

Install the project once, then run one study by ID from the repository root:

```powershell
python -m pip install -e .
python scripts/run_research_experiment.py E01
```

Replace `E01` with an ID below. The default is the existing ten seeds in their
original order. Add `--extended-seeds` to append seeds 10 through 29. Each run
stores its config snapshot, episodes, and summary in
`results/experiments/<ID>/run_<timestamp>/`. Report-ready episode/summary CSV,
LaTeX, and overview PNG/PDF are written to the corresponding
`reports/experiments/<ID>/run_<timestamp>/` directory. Reports can be
regenerated from stored JSON with:

```powershell
python scripts/make_research_reports.py results/experiments/E01/run_<timestamp>
```

Run all twelve studies with up to `parallel_workers` processes from the shared
study config (six by default):

```powershell
python scripts/run_all_research_experiments.py
```

The launcher stores a live `progress.json`, the original per-study config
snapshots, and one log per experiment under
`results/experiments/all_<timestamp>/`.

Each results run folder includes `config_used.json` (the complete resolved
study configuration) and `experiment_config.yaml` (the experiment-specific
settings). Both the results run folder and report folder contain a
`visualizations/` directory with experiment-specific PNG and PDF figures.
These plots are regenerated from `episodes.json` and `summary.json`, so the
report command also restores them if needed.

| ID | Config | Runner | Study |
| --- | --- | --- | --- |
| E01 | `configs/experiments/e01_message_conditioned.yaml` | `e01_message_conditioned.py` | Candidate-fact-conditioned versus unconditioned particles; VoC MAE, receiver/pair regret, and continuation return. |
| E02 | `configs/experiments/e02_role_only.yaml` | `e02_role_only.py` | Static message/status by receiver-role compatibility against aware/agnostic selectors, including an aware-budget-matched role-only arm. |
| E03 | `configs/experiments/e03_particle_diagnostics.yaml` | `e03_particle_diagnostics.py` | Particle-count sweep, duplicate-state diversity ESS, degeneracy, MAE, and regret. The ESS definition is recorded in the output. |
| E04 | `configs/experiments/e04_noise_crn.yaml` | `e04_noise_crn.py` | Rollout-count and common-random-number ablation against a high-count reference, with SE, bias, and runtime. |
| E05 | `configs/experiments/e05_continuation_bias.yaml` | `e05_continuation_bias.py` | Disabled, fixed-frequency, and aware rollout continuations compared with the standard disabled continuation. |
| E06 | `configs/experiments/e06_robust_null.yaml` | `e06_robust_null.py` | Strict positive VoC, 90% lower-bound, and epsilon rules; redundant-message null false positives and H4 cost-rate drops. |
| E07 | `configs/experiments/e07_cost_pareto.yaml` | `e07_cost_pareto.py` | Fine lambda sweep, reward-unit c0, optional receiver-distance cost, and return/message Pareto frontier. |
| E08 | `configs/experiments/e08_hypothesis_robustness.yaml` | `e08_hypothesis_robustness.py` | H1 threshold curves, configured delayed-value H3 cases, H2 exact-budget comparisons, paired intervals, and Holm adjustment. |
| E09 | `configs/experiments/e09_horizon_sweep.yaml` | `e09_horizon_sweep.py` | H=1, 3, 5, 10, 15 agreement, VoC MAE, and context-continuation return. |
| E10 | `configs/experiments/e10_staleness.yaml` | `e10_staleness.py` | Information-age survival heuristic versus offline observed marginal value, with Spearman correlation. |
| E11 | `configs/experiments/e11_robustness.yaml` | `e11_robustness.py` | Paired aware-minus-agnostic return under hazard dynamics, grid-size, and supported team variants. |
| E12 | `configs/experiments/e12_failures_runtime.yaml` | `e12_failures_runtime.py` | Failure contexts, five saved example traces, and particle/rollout/oracle timing breakdown. |

All studies use offline oracle access only for evaluation. Action choices use
local `Observation`, messages, and public role information. Existing default
commands and result folders are separate and remain unchanged. The study
configs use 10,000 bootstrap resamples; set a smaller `bootstrap_samples` in
an experiment-specific config only for a clearly labeled development run.
