# Audit of Receiver-Aware Value of Communication for Cooperative Multi-Agent Search-and-Rescue

## 1. Executive summary

The repository implements and executes a heterogeneous SAR simulator, deterministic local task policies, a paired simulation-based communication selector and offline oracle diagnostics. The strongest evidence is **H4 cost sensitivity**. At the configuration matching the current default, the aware non-myopic policy’s positive-VoC send rate dropped from 15.61% to 1.70%, a relative reduction of 89.11% (paired seed bootstrap 95% CI 75.23%–95.33%). **H1 misses its 30% target** in that configuration. **H2 and H3 are inconclusive** under the specification. The equal-message-count subset does not establish an aware return advantage, and the main experiment pools all contexts for a hypothesis that specifies delayed-value cases.

Two core method discrepancies affect interpretation. The particle estimator samples equiprobable reconstructed worlds without a sequential weighted filter. The agnostic policy assigns one score per message by averaging receiver-conditioned estimates, so the score still depends on actual receiver roles, beliefs and roster. The existing results support a narrower claim about these specific heuristic policies in this simulator.

The specification is all six pages of [final.pdf](final.pdf), especially §§3–6. The separate 19-page proposal, README, plans, plots and notes provide context and result pointers. For headline episode results this audit uses [the default-matching run](results/run_20261121_181949_360025/config_used.json) and treats the fast-hazard configuration featured in the existing notes as a sensitivity case. No registration record identifying a single confirmatory run was found. This audit changes only its two deliverables. [Machine-readable evidence](AUDIT_SUMMARY.json)

## 2. Repository architecture and implementation overview

| Path | Responsibility |
| --- | --- |
| `src/voc_sar/cli.py`, `__main__.py`, `configs/default.yaml` | CLI, YAML configuration and main runner entry. |
| `environment/sar_grid.py`, `types.py` | Layout, movement, local visibility and memory, message delivery, stochastic hazard dynamics, event reward and terminal state. |
| `policies/task.py` | Local Scout exploration and scan, Medic stabilization and disabling, Carrier rescue, candidate fact and intention generation. |
| `communication/selector.py`, `estimator.py` | Message/receiver filtering, reconstructed worlds, paired estimates, cost and send/no-send rule. |
| `evaluation/oracle.py`, `metrics.py` | Privileged counterfactual simulation and metric/bootstrap functions. |
| `experiments/runner.py`, `visualizations.py` | Seed-by-condition episode scheduling, diagnostics, JSON, graphs and sample GIFs. |
| `research_experiments/`, `configs/experiments/`, `scripts/` | E01–E12 context analyses, batch launch/progress, report generation and gamma/lambda sweep. |
| `results/`, `reports/`, `analysis_final/` | Primary episode rows and summaries, exploratory context rows and logs, derived CSV/TeX and plots, synthesis notes. No learned model checkpoints surfaced. |
| `tests/`, `.github/workflows/ci.yml`, `pyproject.toml` | Unit/smoke tests, CI configuration and package metadata. Working tree contains pre-existing edits and untracked study outputs. |

Paths in the next two sections are relative to `src/voc_sar/` unless prefixed with `tests/` or `configs/`. Status distinguishes spec alignment from execution and statistical evidence.

## 3. Implementation completeness matrix

| Component | Implemented | Test evidence | Execution evidence |
| --- | --- | --- | --- |
| Simulator and matched task policies | Yes, including local policies and reward. Transport is abstracted. | Core seeded layout, local boundary, transition and reward tests. | Main episode rows and 20 independently reproduced Never episodes. |
| Aware and agnostic valuation | Both selectors exist. Agnostic still uses receiver-conditioned predictions. | Aware routing and role-free agnostic *shortlisting* tests. | Main policy episodes and exploratory offline approximations. |
| Particle belief engine | The 64-world approximation exists. Sequential weighted filtering is absent. | Sender-local prior and sample conditioning checks. | Main runs and E01/E03. |
| Paired rollouts and offline oracle | Yes, matches rollout structure with a send-branch initialization caveat. | Core paired/oracle test and CRN/disabled continuation equivalence tests. | Main diagnostics and E04/E05. |
| Baselines and ablations | Four primary baselines, role-only and E01–E12. | Small analyzer/report smoke tests. | Nine main run folders, legacy top-level mirror, two batches and sweep. |
| Matched-budget and delayed-value protocols | Post-hoc equal counts and an exploratory delayed analysis exist. | Metric helpers and smoke tests. | No prospective budget experiment or full primary delayed-value diagnostic recovered. |
| Secondary and stretch work | Receiver-dependent cost has an E07 option. E11 varies grid size, hazards and team composition. | Helper and analyzer tests. | Base E07 distance cost is off. CMV, TarMAC, learned surrogate and multi-sender result evidence absent. |

## 4. Report-to-code alignment matrix

| ID | Requirement | Status | Code evidence | Implication and correction |
| --- | --- | --- | --- | --- |
| A1 | Three roles and deterministic local policies | ALIGNED | environment/sar_grid.py:96; policies/task.py:17–73 | Ship fixed policy versions with run manifest. |
| A2 | Victim progression and role actions | PARTIAL | environment/sar_grid.py:392–443 | Carrier rescue is adjacent and immediate. Implement transport or describe the abstraction. Record hazard/assignment semantics explicitly. |
| A3 | Shared reward and discounted returns | ALIGNED | environment/sar_grid.py:350–376; communication/estimator.py:149–183; experiments/runner.py:332–343 | 20 Never replays agree exactly. Persist step reward and event traces for all methods. |
| A4 | Episode termination and matched task policies | ALIGNED | environment/sar_grid.py:367–376; experiments/runner.py:193–218,320–330 | State the all-rescued early exit and its impact on exposure. |
| B1 | Receiver identity, role and summary | PARTIAL | communication/selector.py:44–92,124–152; environment/sar_grid.py:204–224,517–540 | Role determines simulated behavior. Sender's own deliveries only approximate receiver memory. |
| B2 | Weighted 64-particle receiver filter | PARTIAL | communication/estimator.py:29–57,110–126; environment/sar_grid.py:104–224 | 64 equiprobable reconstructed worlds are used. Validate against known states and implement a temporal weighted update or revise the specification. |
| B3 | Receiver observations and memory | PARTIAL | environment/sar_grid.py:104–224,477–540; communication/estimator.py:149–166 | Approximate other agents' local histories and distinguish stale tracked positions from current observations. |
| B4 | Net VoC | ALIGNED | communication/estimator.py:137–146,190–196; configs/default.yaml | Primary messages have unit cost. Document optional per-message cost metadata. |
| C1 | Receiver-independent agnostic score | MISALIGNED | communication/selector.py:156–187 | Use a frozen receiver-blind reference and verify invariance to actual receiver state, role and receiver roster at fixed local message features. |
| C2 | Seeded fixed receiver selection | ALIGNED | communication/selector.py:163–167; experiments/runner.py:251 | Use the same seeded routing rule in offline policy reproductions. |
| D1 | H=1/5 online, H=15 offline oracle | PARTIAL | experiments/runner.py:56–67,386–505; research_experiments/common.py:195–212; research_experiments/e09_horizon_sweep.py:8–22 | Preserve the main configuration and label offline H=15 estimator sweeps separately. Two main runs use oracle H=35. |
| D2 | Truncated finite-horizon return | ALIGNED | communication/estimator.py:149–183; environment/sar_grid.py:367 | Keep same terminal semantics in simulated states and observed episodes. |
| E1 | Paired clones, CRN, same policies, no subsequent communication | ALIGNED | communication/estimator.py:114–146,149–183; evaluation/oracle.py:43–89; environment/sar_grid.py:448–475 | Add branch-marginal and state-equivalence checks. |
| E2 | Candidate delivery as sole intervention | PARTIAL | communication/estimator.py:156–166; environment/sar_grid.py:277–301 | Separate validation from sender-memory mutation; test exact branch state changes. |
| E3 | Unbiasedness for real receiver-conditioned return | UNVERIFIED | environment/sar_grid.py:104–224; research_experiments/e04_noise_crn.py:12–76 | Evaluate model error separately from finite Monte Carlo error. |
| F1 | Offline oracle, decentralized valuation and one active sender | ALIGNED | experiments/runner.py:193–258,315–321,386–478; evaluation/oracle.py:43 | Automate end-to-end privilege checks. |
| F2 | Test rejects global-state reads along decentralized path | PARTIAL | tests/test_core.py:89–127,185–223 | Add a runner-level guard. Existing tests cover observation shape, local particle priors and oracle misuse. |
| G1 | Directed message cap, candidate kinds, pre-delivery decisions | ALIGNED | policies/task.py:76–94; environment/sar_grid.py:277–301; experiments/runner.py:230–325 | Enforce pre-delivery validation as a batch if adding multi-sender runs. |
| G2 | Greedy positive-VoC selection | PARTIAL | communication/selector.py:124–187; experiments/runner.py:511–543 | Reuse the online selection/tie rule in evaluation and handle empty agnostic shortlist. |
| X1 | Never, Random, Full, Fixed-frequency | ALIGNED | experiments/runner.py:56–67,264–313 | Label Full as one heuristic-ranked message per eligible step, Random as randomized content/routing, and Fixed as every third step. |
| X2 | Matched computation and communication budgets | MISALIGNED | communication/selector.py:135–173; experiments/runner.py:758–826 | Equalize pair evaluations and enforce prespecified budgets prospectively. |
| X3 | Ten-seed evaluation and uncertainty | PARTIAL | experiments/runner.py:69–79,627–681,683–863 | Primary configurations have ten seeds. Persist seed/context data and test hypotheses on their stated estimands. |
| X4 | CMV, TarMAC, learned surrogate, multi-sender tests | NOT_IMPLEMENTED | experiments/runner.py:56–67; configs/experiments/study_base.yaml | Treat secondary references and stretch work separately. |
| X5 | Staleness validation and Pareto | PARTIAL | communication/selector.py:72–88; research_experiments/e07_cost_pareto.py:29–91; research_experiments/e10_staleness.py:9–43 | Use age variation and full-episode budget/cost arms. |

Additional protocol details:

- **Task semantics.** The Scout scan marks hidden victims located and discovers hazards. Any role can subsequently observe visible, already-located victims. Medic disables a nearby active hazard for a configurable duration. Carrier increments rescue at adjacency with no pickup, transit or drop-off. Environment state tracks positions, victims, hazards, memory and timers. There is no explicit assignment subsystem corresponding to the report’s state notation. Reward charges all three agents per tick including idle and failed actions, and communication is excluded. Episodes stop at the cap or after all victims are rescued by default. [Simulator](src/voc_sar/environment/sar_grid.py)
- **Belief semantics.** The sampler starts from a seeded reset and rebuilds worlds using sender facts, tracked positions and uniform placement of missing entities. It hardens stale tracks and facts, approximates unknown hazard state and duration, and uses sender-specific delivery history as receiver memory. The local observation model is deterministic conditional on the sampled state and simulated histories, but only the sender’s visitation and scanning history are restored. Unknown positions may be sampled in sender-visible cells and negative-evidence conditioning is incomplete. A uniform sample supports a distribution over reconstructed environment states. The implementation supplies no particle weights beyond uniform sampling, likelihood update, temporal propagation of a posterior, or resampling. [Sampler](src/voc_sar/environment/sar_grid.py#L104) · [Estimator](src/voc_sar/communication/estimator.py#L29)
- **Counterfactual estimand.** Two clones start from the same reconstructed particle or actual offline context and receive the same dynamics seed. The simulator draws three uniforms per hazard per step before applying branch-specific state effects. Task policies are deterministic, continuation has no communication and return discounts from the intervention time through the earlier of horizon or terminal state. Across a correctly sampled initialization distribution and correct branch marginals the paired mean targets the model’s expected send-minus-no-send return. Approximate beliefs/history, forced sender-fact merge, finite oracle precision and a maximum over noisy scores all limit interpretation. CRN changes variance through covariance. It does not establish unbiasedness, and E04 compares against a larger sample from the same model whose CRN prefix overlaps. [Estimator](src/voc_sar/communication/estimator.py#L95) · [Oracle](src/voc_sar/evaluation/oracle.py#L43) · [Hazard dynamics](src/voc_sar/environment/sar_grid.py#L448)
- **Computation and baselines.** Aware prices up to eight pairs. Agnostic prices up to eight messages for each receiver, up to 16 pairs in the three-agent case. Both get 32 rollouts per pair. Random sends a random message to a random receiver on every eligible step. Full always sends one cheaply ranked pair; Fixed-frequency uses the same rank every third step. Main baseline cost entries are zero. Under the stated pre-delivery protocol only Scout is active in normal episodes and diagnostics, and the oracle has no online call site. [Selector](src/voc_sar/communication/selector.py#L95) · [Runner](src/voc_sar/experiments/runner.py#L181)

## 5. Experiment execution summary

All main rows have unique (method, lambda, seed) keys, complete configured seed sets, finite numeric scalars and consistency between nonzero-denominator reported rates and counts. A seed contributes one episode to each of ten conditions (four baselines plus three estimated policies at two costs), yielding 10 episodes per seed **across conditions**. No 100-episode-per-seed run is evidenced. Main diagonal comparisons use identical task policy code and environment seeds across methods. The layout seed is reset once per condition/seed, intentionally matching that replicate across methods. Runs with identical non-timing episode data are flagged below and in the JSON to avoid counting repeat runs as additional seeds.

All listed main runs configure 64 particles, shortlist 8, 32 paired rollouts per evaluated pair, lambda 0/0.5, unit cost 1, H=1/5 online, cap 100 steps, no valuation-context cap, hazard toggle 0.2 and three roles. Grid-10 runs use 3 victims and 2 hazards. Saved full configurations, including the grid-30 overrides, are in the JSON. No equal episode message cap was assigned. The single-Sender per-step cap is 1. Main actual valuation counts by condition are in the JSON.

| Stored run | Seeds / episodes | Grid, gamma, move probability | Oracle H | Diagnostic contexts | H1 hits / eligible | Positive exact-budget seeds | H4 drop |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| results | 10 / 100 | 10, 0.99, 0 | 15 | 100 | 17/53 | 7 | 81.48% |
| results/run_20261113_124225_572171 | 10 / 100 | 10, 0.99, 0 | 15 | 100 | 17/53 | 7 | 81.48% |
| results/run_20261120_210403_230837 | 10 / 100 | 10, 0.99, 0 | 15 | 100 | 17/53 | 3 | 82.96% |
| results/run_20261121_021746_338196 | 20 / 200 | 30, 0.99, 0 | 35 | 100 | 0/10 | 12 | 95.83% |
| results/run_20261121_061856_659878 | 20 / 200 | 30, 0.99, 0 | 35 | 100 | 0/10 | 12 | 95.83% |
| results/run_20261121_120515_271425 | 10 / 100 | 10, 0.99, 0 | 15 | 100 | 17/53 | 3 | 82.96% |
| results/run_20261121_123226_187116 | 10 / 100 | 10, 0.9, 0.8 | 15 | 100 | 14/49 | 7 | 86.00% |
| results/run_20261121_161323_602973 | 10 / 100 | 10, 0.9, 0.8 | 15 | 100 | 14/49 | 7 | 86.00% |
| results/run_20261121_170936_173219 | 10 / 100 | 10, 0.9, 0.2 | 15 | 100 | 14/55 | 3 | 87.35% |
| results/run_20261121_181949_360025 | 10 / 100 | 10, 0.8, 0.2 | 15 | 100 | 6/55 | 5 | 89.11% |

The top-level results pair has the same non-timing episode data as November 13. The November 20 210403 / November 21 120515 pair, 021746 / 061856 pair and 123226 / 161323 pair also have identical non-timing episode data within each pair. They remain single evidence sets for inference. The 30×30 runs and H=35 oracle are exploratory relative to the specified H=15. Stored episode equivalence does not establish originating commit or intent. [Results directory](results)

Both E01–E12 batch manifests record 12 completions and no failures. Two earlier E01 attempts also have full artifact sets. Each additive run listed below records 10 seeds (0–9). Each context is sampled along a no-send trajectory. Tables named `episodes.json` here contain **context/condition records**. E11 collects one context per seed per setting; its summary's 10 is the base collection size and its 90 rows cover nine settings. Stored configuration and script overrides jointly determine effective settings. [Earlier batch](results/experiments/all_20261121_153923_146992/progress.json) · [Later batch](results/experiments/all_20261121_182538_166981/progress.json)

| Study / run | Manifest contexts | Rows | Wall seconds |
| --- | ---: | ---: | ---: |
| E01/run_20261121_151823_224945 | 20 | 40 | 20.5 |
| E01/run_20261121_153524_115520 | 20 | 40 | 33.0 |
| E01/run_20261121_154000_756858 | 20 | 40 | 37.3 |
| E01/run_20261121_182611_236195 | 20 | 40 | 32.6 |
| E02/run_20261121_153948_887162 | 20 | 80 | 25.4 |
| E02/run_20261121_182600_198353 | 20 | 80 | 21.6 |
| E03/run_20261121_154003_982133 | 20 | 100 | 40.5 |
| E03/run_20261121_182612_760521 | 20 | 100 | 34.1 |
| E04/run_20261121_154150_781536 | 20 | 200 | 147.3 |
| E04/run_20261121_182739_910360 | 20 | 200 | 121.3 |
| E05/run_20261121_154540_962162 | 20 | 60 | 377.5 |
| E05/run_20261121_183115_027376 | 20 | 60 | 336.4 |
| E06/run_20261121_154236_875337 | 20 | 120 | 193.4 |
| E06/run_20261121_182808_270757 | 20 | 120 | 149.6 |
| E07/run_20261121_154035_483010 | 20 | 720 | 45.9 |
| E07/run_20261121_182638_142372 | 20 | 720 | 37.2 |
| E08/run_20261121_154050_919576 | 20 | 404 | 49.6 |
| E08/run_20261121_182652_564758 | 20 | 404 | 40.7 |
| E09/run_20261121_154143_990937 | 20 | 100 | 99.5 |
| E09/run_20261121_182735_053944 | 20 | 100 | 81.5 |
| E10/run_20261121_154048_396393 | 40 | 40 | 12.1 |
| E10/run_20261121_182649_059693 | 40 | 40 | 10.2 |
| E11/run_20261121_154128_600452 | 10 | 90 | 39.7 |
| E11/run_20261121_182723_737946 | 10 | 90 | 34.1 |
| E12/run_20261121_154203_586664 | 40 | 40 | 72.0 |
| E12/run_20261121_182749_519315 | 40 | 40 | 56.3 |

| Study | Effective execution and interpretation |
| --- | --- |
| E01 | Base vs message-conditioned worlds, 64/32. Context continuations, same selected state-support assumptions. |
| E02 | Aware, agnostic, role-only and role-only constrained to aware’s context send flag. 64/32. Reimplements routing and uses offline continuation. |
| E03 | Particle counts 16/32/64/128/256 at 32 rollouts and H=15. Single candidate per context. |
| E04 | CRN on/off, 8/16/32/64/128; 256-rollout CRN reference, 64 particles, H=15. |
| E05 | Disabled, fixed-frequency and aware continuation, 32 outer pairs. Nested aware uses 8 particles, 4 rollouts, shortlist ≤3. Targets change with background policy. |
| E06 | Strict/LCB/epsilon rules, lambda 0/0.5. 100 null draws/context with 8 independent-branch rollouts each and 32 CRN pairs for real messages. |
| E07 | Six lambdas from 0 to 1 × six methods, 64/32, first three messages. Unit cost 1, distance cost disabled. One starting intervention per context. |
| E08 | 20 contexts, thresholds grid, delayed margin 0.5. H=1/5 estimates and H=1/15 oracle for delayed analysis. H2/H4 use context intervention calculations. |
| E09 | H=1/3/5/10/15 estimator sweep against H=15 oracle, 32 rollouts, first three messages. |
| E10 | 40 contexts, age bins configured. All 40 saved ages are 0. |
| E11 | Variable grid, hazard dynamics and 2–3 roles, 16-rollout pair valuation. Grid size change is implemented, independent layout-family evaluation remains absent. |
| E12 | 40 contexts, selected-pair timing and failure traces, 64/32. |
| Gamma/lambda sweep | Four gammas (0.8/0.9/0.95/0.99) × six lambdas × seven methods, 168 CSV summary rows. Code requests two contexts per seed, 64/32, H=15 default in `evaluate_pairs`. Aggregate JSON/CSV only. Seed/context outcomes unavailable for independent verification. |

## 6. H1–H4 hypothesis results

The audit recomputed main H2/H4 from raw episode rows using condition/seed pairing and a 10,000-draw seed percentile bootstrap (RNG seed 215). For H1 it thresholded the stored ordered differences and reconstructed the seed partition from the configured no-send context trajectories. No historical message or context identity is saved, so that partition is conditional on the current code reproducing historical context selection. All configurations remain separate.

### H1 · FAIL in the default-matching environment

At gamma 0.8 and hazard move 0.2, **6/55 eligible contexts = 10.91%**. The seed-cluster interval is **[0.0%, 27.87%]** with the reconstructed partition. Stored CI is [0.0%, 28.81%]. The remaining 45 of 100 held-out decision contexts had no non-intention fact under the replay. Code chooses the first non-intention candidate and evaluates that exact item for both receivers on cloned versions of the same world. Archived individual receiver values, message IDs and world IDs are **UNVERIFIED**. Save these alongside candidate ordering and oracle MC uncertainty.

| Seed | Default matches over eligible | Fast-hazard matches over eligible |
| ---: | ---: | ---: |
| 0 | 0/9 | 0/9 |
| 1 | 0/0 | 0/0 |
| 2 | 5/9 | 7/9 |
| 3 | 0/7 | 0/0 |
| 4 | 0/5 | 0/7 |
| 5 | 1/3 | 1/3 |
| 6 | 0/5 | 0/7 |
| 7 | 0/3 | 0/5 |
| 8 | 0/5 | 0/0 |
| 9 | 0/9 | 6/9 |

Fast-hazard sensitivity gives **14/49 = 28.57%**, audit CI [2.04%, 54.90%], so that setting is inconclusive relative to 30%. The legacy 17/53 point estimate clears 30% with a wide interval containing it. E08's primary 1/20 row count additionally requires `peak > 0.5`, with a different context definition and seed reuse. Its p-value cannot be paired with the primary run’s 14/49 figure. [Main diagnostic source](src/voc_sar/experiments/runner.py#L553) · [E08 source](src/voc_sar/research_experiments/e08_hypothesis_robustness.py#L39)

### H2 · INCONCLUSIVE

- **Matched return arm.** At lambda 0, Aware H=5 has mean J0 **1.2252** and **120** messages versus Full **1.1680** and **897**. All 10 seed pairs are within the stored ±1.0 return match tolerance. The specified ratio of totals gives **86.62%** reduction. Averaging per-seed ratios gives the stored 85.84%. The descriptive requirement of 20% is met under the run’s tolerance. Selecting pairs by realized return and tolerating a 1.0 difference at gamma 0.8 do not establish equivalence. Predefine a meaningful margin and infer the paired return difference/equivalence on an independent seed set.
- **Matched communication arm.** Exact positive message counts occur on **five seed/cost pairs** in the default-matching run. All five paired return differences are zero and the bootstrap CI is **[0, 0]**. Two additional zero/zero comparisons are also ties. The fast-hazard sensitivity has seven positive equal-count seed/cost pairs with difference +0.00135 and CI [0, 0.00405]. The required CI exclusion of zero fails in both. Although the unrestricted paired differences are +0.1171 [0.0294, 0.2246] and +0.5724 [0.1622, 1.0792] respectively, the budgets differ materially. Selectivity against the agnostic policy remains unestablished. A prospective fixed-budget experiment is **NOT EXECUTED**.
- **Arm-specific comparisons.** At default lambda 0, Agnostic has mean return **1.1001** and **326** messages. At 0.5 it has **0.9754** and **78** messages. E08 only matches realized 0/1 context interventions and uses a single continuation trajectory per chosen action. Its unmatched sign-flip p-value cannot be used for H2. Use a shared prospective budget controller, a receiver-blind value ablation, equal computation and explicit handling of unspent capacity. [H2 implementation](src/voc_sar/experiments/runner.py#L758) · [Episode rows](results/run_20261121_181949_360025/episodes.json)

### H3 · INCONCLUSIVE

Section 5 of the PDF specifies **delayed-value cases**. Primary summary gives **74/100** H=1 and **79/100** H=5 agreement against H=15, a gain of **+0.05** with stored seed-cluster CI **[−0.23, 0.34]**. Fast-hazard gives **77/100**, **81/100**, gain **+0.04** with CI **[−0.20, 0.32]**. These are reported aggregates, not independently recoverable action counts. The main delayed subset, action triples, and seed-level agreement are **UNVERIFIED** from archives. Even the all-context diagnostic reconstructs decisions with a different tie rule: on positive ties it prefers larger pair index, while the online aware selector prefers the first ranked pair.

E08 contains four delayed rows, selected if any candidate’s oracle absolute difference between H=15 and H=1 exceeds 0.5. The per-context gains for (seed, step) are (3,0) 0, (4,0) 0, (9,0) −1, (9,1) +1. Mean gain 0 and three-seed percentile interval [0,0] reflect opposing effects and limited support, not precise population equivalence. E08 stores only the agreement difference. E09 independently stores binary agreements over the same 20 seed/step keys. All-context E09 gives H=1 **90%**, H=5 **75%**, paired gain **−0.15** [−0.35, 0.05]. An exploratory join to the E08 delayed keys gives **75%** and **25%** (−0.50) from four contexts. E08 and E09 seed their estimators differently and neither uses the full online candidate selector. The joined result measures E09 decisions on those keys. E09 per-seed all-context H=1/H=5 agreements (two contexts each) are 0 1/1, 1 1/1, 2 1/1, 3 1/.5, 4 1/.5, 5 1/.5, 6 .5/1, 7 1/.5, 8 1/1, 9 .5/.5. In the delayed join, seeds 3 and 4 each give 1/0, and seed 9 gives .5/.5. Confirmatory delayed-gain evidence remains missing. [Main diagnostic](src/voc_sar/experiments/runner.py#L511) · [E08](src/voc_sar/research_experiments/e08_hypothesis_robustness.py#L50) · [E09](src/voc_sar/research_experiments/e09_horizon_sweep.py#L8)

### H4 · PASS for the evaluated policy and setting

| Configuration | Lambda 0 rate / count / mean return | Lambda 0.5 rate / count / mean return | Relative reduction [95% CI] |
| --- | --- | --- | --- |
| Default-matching | 15.61% / 120 / 1.2252 | 1.70% / 17 / 1.0846 | 89.11% [75.23%, 95.33%] |
| Fast-hazard sensitivity | 16.66% / 126 / 3.4361 | 2.33% / 20 / 2.8549 | 86.00% [70.32%, 94.14%] |

Each seed rate is sends divided by eligible valuation decisions. Average the seed rates, then take one minus the high/low ratio. The audit bootstraps that same ratio on paired seed draws. In the default run the denominators total 906 and 1,000. No seed has a zero low rate in either table row. For a replication with an all-zero low rate the relative decrease is undefined; record the zero count and report the absolute change. The main code instead returns zero in that case and its interval averages per-seed ratios, omitting seeds with low rate zero. E06’s fixed-candidate rates are exploratory. A drop on a frozen score surface follows mechanically from raising the threshold; the online result also reflects path and exposure changes and supports the directional behavioral prediction for this tested setup. [H4 implementation](src/voc_sar/experiments/runner.py#L828) · [Raw recalculations](AUDIT_SUMMARY.json)

## 7. Metrics independently recalculated from raw data

### Default-matching episode aggregates

Means and sample SDs are over ten seeds per condition. Counts are sums, and decisions use the runner's method-specific definition. The JSON includes all runs, condition-level dispersion for both return and messages, measured episode times, rescue totals, and full seed-pair values for the primary contrasts. These times include simulator/policy work and, on capture jobs, visualization frame collection. Wall-clock episode times also reflect concurrent workers.

| Condition | Mean J0 ± SD | Messages summed | Mean messages | Recorded decisions | Episode seconds mean ± SD |
| --- | ---: | ---: | ---: | ---: | ---: |
| never@0 | 0.9077 ± 1.7466 | 0 | 0.0 | 1000 | 0.0248 ± 0.0024 |
| random@0 | 0.8443 ± 1.6513 | 908 | 90.8 | 908 | 0.0261 ± 0.0070 |
| full@0 | 1.1680 ± 1.9184 | 897 | 89.7 | 897 | 0.0285 ± 0.0091 |
| fixed_frequency@0 | 1.1326 ± 1.8813 | 316 | 31.6 | 316 | 0.0290 ± 0.0076 |
| aware_myopic@0 | 0.9077 ± 1.7466 | 0 | 0.0 | 1000 | 65.8288 ± 3.3441 |
| aware_nonmyopic@0 | 1.2252 ± 1.8368 | 120 | 12.0 | 906 | 108.7760 ± 34.0465 |
| agnostic_nonmyopic@0 | 1.1001 ± 1.9046 | 326 | 32.6 | 1000 | 179.7957 ± 25.0887 |
| aware_myopic@0.5 | 0.9077 ± 1.7466 | 0 | 0.0 | 1000 | 72.2314 ± 3.5290 |
| aware_nonmyopic@0.5 | 1.0846 ± 1.8748 | 17 | 1.7 | 1000 | 126.4648 ± 8.5746 |
| agnostic_nonmyopic@0.5 | 0.9754 ± 1.7985 | 78 | 7.8 | 1000 | 174.1361 ± 28.4647 |

| Metric | Recalculation or limitation | Audit consequence |
| --- | --- | --- |
| J0 | 20 Never seed/config pairs replayed with local policies, then reconstructed as the sum of discounted rewards from event counts and three charged actions per tick. Maximum absolute stored discrepancy 0. | Original per-step reward and action rows are absent for other methods. Their historical aggregate J0 is observed, step-level reconstruction **UNVERIFIED**. |
| Message, rate and CRB | Recomputed from persisted episode counts. Main CRB uses average of per-seed ratios, which differs from report ratio of summed message counts. | Use aggregate ratio as primary and explicitly label secondary per-seed estimates. See H2/H4. |
| Receiver regret | Main summary reports 0.000160 in default run. Code uses a clipped comparator including no-send for the primary message; choosing a different message counts as no-send for that comparison. | PDF Eq. 25 compares receivers for the same message. Report conditional receiver regret and separate no-send decisions. Pair-level recomputation **UNVERIFIED**. |
| Pair regret | Main reports 0.010328; E12 row-level regret averages to 0.32998 in its own 40 contexts. Eq. 26 implementation clips against the best positive alternative including no-send. | Main shortlist/choice/oracle arrays unavailable. E12 row regret is for the best pair selected across candidates while its adjacent trace fields describe a separately sampled candidate. |
| VoC MAE | Main reports 0.113618. It pools all shortlisted H=5 predictions versus H=15 oracle, rather than equal weighting of contexts, at zero cost. E03 64-particle H=15 same-horizon row MAE averages 0.59445. | Main number combines model and truncation errors and weights by candidate count. Paired inputs **UNVERIFIED**. Report same-H and cross-H errors separately. |
| Filter recall | Main reports 1.0; helper returns 1 when there are no above-threshold pairs and the runner averages contexts. | Number of contexts with any high-value pair, non-vacuous recall, and aggregate pair recall **UNVERIFIED**. Report both coverage and recall conditional on an eligible denominator. |
| Useful-message rate | Main reports 0.55814, among H=5 sends in offline diagnostics, oracle MV > 0. | Selected counts and per-message MV are **UNVERIFIED** in main archives. Distinguish diagnostic usefulness from online send usefulness. |
| Runtime | E12 row means 0.01367 s particle sample, 0.10404 s approximate rollout component and 0.23350 s oracle, per sampled pair. E04 32-rollout CRN timing averages 0.25259 s for its H=15 loop. | E12 subtracts a separate sampling measurement from the full estimator measurement and clips at zero; E04 excludes particle setup. Neither measures complete selection latency. Main run wall time is 1,365.89 s and episode timing is directly available. |
| Pareto | Recomputed non-dominated conditions from mean J0 and mean messages. Default frontier is zero-message Never / both myopic ties, then aware H=5 at 0.5 and 0. E07 frontier recomputed from 720 rows and included in JSON. | E07 uses messages 0/1 and return from the context to terminal. Present its points as context interventions. Main frontiers have a coarse cost grid and no uncertainty bands. |
| Horizon agreement | E09 rows independently aggregated and paired by seed/step, including the E08 delayed-key join in §6. | Actual main action identity and per-seed H3 **UNVERIFIED**. Separate send/no-send agreement from recipient/content agreement and publish no-send prevalence. |

Metric implementations are in [metrics.py](src/voc_sar/evaluation/metrics.py), [main diagnostics](src/voc_sar/experiments/runner.py#L386), [main summaries](src/voc_sar/experiments/runner.py#L683) and [E12 timing](src/voc_sar/research_experiments/e12_failures_runtime.py#L11).

### Exploratory row checks

- E03 fingerprint ESS fraction is 1.0 at each particle count and the recorded degeneracy rate is zero. This statistic is inverse collision frequency of coarse fingerprints; it does not measure importance weight concentration or posterior accuracy. Mean regret is 0.54050, 0.54050, 0.24169, 0.29882 and 0.29882 across increasing particle counts. One setting and 20 contexts give weak evidence for an optimal count. [E03 analyzer](src/voc_sar/research_experiments/e03_particle_diagnostics.py#L11)
- E04 recorded mean SE is 0.08174 with CRN versus 0.11190 without at 32 rollouts, and 0.03600 versus 0.05716 at 128. E04's “bias” is a signed discrepancy from a finite same-model reference. The audit verifies the average of saved per-context SEs; the raw branch returns are absent. [E04 analyzer](src/voc_sar/research_experiments/e04_noise_crn.py#L12)
- E06 recorded strict rates are 2/20 vs 0/20, LCB rates 1/20 vs 0/20, and epsilon 0.5 is 0/20 at both costs with undefined relative drop. Null false-positive rates are summary-only. The null protocol uses independent branches to induce variation even for irrelevant messages, while operational estimates use CRN. This demonstrates behavior under an artificial null/noise condition. Store null trials and test naturally redundant valid messages with the actual shortlist and CRN. [E06 analyzer](src/voc_sar/research_experiments/e06_robust_null.py#L23)
- E10 ages are all 0 and average recorded oracle MV is −0.42583 across 40 contexts. Age cannot explain variation in this sample. E12 likewise records age 0 throughout and its exploratory mean return contrast is −0.26432 against its agnostic variant and −0.39226 against role-only. [E10 analyzer](src/voc_sar/research_experiments/e10_staleness.py#L9) · [E12 analyzer](src/voc_sar/research_experiments/e12_failures_runtime.py#L28)
- E01 conditional vs base means are equal in saved rows. E01 forces facts into particles that are already conditioned on sender facts, with limited scope to change state under these fresh contexts. E05's continuation variants estimate different policy interventions and use a reduced nested selector. E11 setting effects remain local to one early context per seed per setting. Detailed group means and configurations are in the JSON. [E01 analyzer](src/voc_sar/research_experiments/e01_message_conditioned.py#L27) · [E05 analyzer](src/voc_sar/research_experiments/e05_continuation_bias.py#L63) · [E11 analyzer](src/voc_sar/research_experiments/e11_robustness.py#L50)

## 8. Statistical validity assessment

- **Unit of inference.** Main episode pairs use seeds. Two costs on the same seed form a cluster. In held-out trajectories, contexts within a seed share state progression and correlated outcomes. The runner uses seed clusters for H1 and all-context H3, appropriate for within-seed dependence as a first approximation. The H1 bootstrap weights contexts within resampled clusters and conditions on eligible seeds. Ten seeds give weak tail precision for several hypotheses; thousands of bootstrap draws reduce numerical error without adding experimental replication. H3's three delayed seeds offer particularly poor support. The gamma/lambda script bootstraps context-level paired differences and therefore underrepresents clustering. [Cluster bootstrap](src/voc_sar/evaluation/metrics.py#L102) · [Sweep CI](scripts/sweep_gamma_lambda.py#L164)
- **Estimands and tests.** H1 should assess 30%, H2 should use both arms and the exact stipulated CI criterion, H3 should assess a 0.10 gain on delayed contexts and H4 should assess a 0.15 relative drop. Main `significance_level` is configuration metadata and main flags often use point estimates. E08 sign flips a mean of seed effects. H1 uses an extra gate, H2 the unrestricted contextual contrast, H3 zero gain and H4 absolute difference from zero. Its Holm arithmetic adjusts those p-values, which address different quantities from the four PDF targets. Sign flipping also requires an exchangeability/symmetry justification and is very discrete at small seed counts. Avoid interpreting incompatible p-values and confidence intervals as simultaneous confirmation. [H1–H4 flags](src/voc_sar/experiments/runner.py#L883) · [E08 tests](src/voc_sar/research_experiments/e08_hypothesis_robustness.py#L92) · [Statistical helpers](src/voc_sar/research_experiments/common.py#L316)
- **Condition selection and failures.** Fast, slow, static and larger-grid settings use overlapping seeds. Aggregate conclusions across replications require seed-aware blocking and a predeclared target distribution. Store all requested jobs and explicit exclusions. Completed batch logs document no batch failures. Failed attempts, excluded contexts and historical potential configuration tuning on the same outcomes remain **UNVERIFIED** where provenance is absent. The main run writes outputs after diagnostics, so interrupted jobs can leave no archived episode record. [Runner persistence](src/voc_sar/experiments/runner.py#L104) · [Existing synthesis](analysis_final/notes/results_summary.md)
- **Return scale.** Changes in gamma alter the estimand, discount delayed rescues and change VoC and policy decisions. Absolute margins such as 1.0 are large relative to some observed J0 values. Publish rescue count and time-to-rescue beside J0 and select a scale-aware equivalence margin in advance. [Main reward aggregation](src/voc_sar/experiments/runner.py#L332)

## 9. Reproducibility and leakage audit

`PYTHONPATH=src python -B -m pytest -p no:cacheprovider -q` (expressed as setting the environment variable in PowerShell) gave **19 passed** outside sandbox restrictions. The sandbox attempt had 16 passed and three integration failures involving Windows process or temporary directory permissions. Tests used temporary outputs. The current suite covers event reward, role restrictions, scan/delivery/transition order, local particle prior, seeded layout and hazard transitions, paired estimator and oracle separation, deterministic metric/bootstrap helpers, runner output and all additive analyzers. The file-based integration test checks row count and configuration recording. It does not compare complete repeated aware episodes. The current-code 20 Never replay checks produced identical repeated reward sequences and matched archived J0 values for these two configurations. A complete historical rerun for valuation methods and commit-to-result identity remains **UNVERIFIED**. [Core tests](tests/test_core.py) · [Study tests](tests/test_research_experiments.py) · [CI](.github/workflows/ci.yml)

The deployed decision frontend receives sender observations, role roster and config. Global data used in dynamics, post-episode counts, rendering and oracle comparisons has no traced online selection path. Sender-provided estimates use sampled model worlds. The existing boundary tests rule out several immediate mistakes, including passing an observation to the oracle and using a true world as particle input, but do not trap every privileged read in the full decentralized execution. Implement a runner-level guarded simulator or perturb hidden state at fixed sender observations and compare policy calls, estimates and choices. [Runner](src/voc_sar/experiments/runner.py#L181) · [Oracle test](tests/test_core.py#L185)

Primary diagnostic layouts use seed +77,777 and a separate no-send collection loop. This gives separation from the evaluated episode trajectories by seed in recorded runs. No trainable model is present. Additive studies reset the same numbered seeds directly, reusing main initial layouts when environment settings match. Those contexts are useful for exploration but do not furnish independent confirmation. Hash config, code and context IDs and publish separate tuning and confirmation manifests. [Primary context generator](src/voc_sar/experiments/runner.py#L423) · [Additive collector](src/voc_sar/research_experiments/common.py#L97)

## 10. Bugs and methodological inconsistencies

| Priority | Finding and evidence | Recommended correction |
| --- | --- | --- |
| High | Receiver-dependent estimates in the “agnostic” score. `select_communication` lines 156–187 calculates each receiver’s value before averaging. E02/E07/E08/E12 substitute their own receiver routing. | Use a frozen receiver-blind scoring model, a matched candidate/compute budget and one common selection implementation. Add invariance tests. |
| High | `_run_held_out_diagnostics` lines 511–543 and `_summarize` lines 937–949 use all contexts and different tie selection for H3. Main aggregate cannot recover delayed context actions. | Log each actual selector action and oracle action with stable message identity, delayed eligibility, inclusion reason and oracle uncertainty. |
| High | `_summarize` lines 758–826 matches realized outcomes. `matched_budget_episode_count` is a seed count after aggregation. E08’s one-intervention subset is another estimand. | Assigned budget protocol plus counts of both paired observations and distinct seeds. Include and identify zero-send pairs. |
| High | `analyze` in E08 lines 39–89 and 112–140 applies different hypotheses. Derived notes/CSV combine H1 main numerator with a different p-value and report H2 unrestricted return under disputed selectivity language. | Regenerate the hypothesis summary from validated held-out rows and a one-to-one mapping between tests and estimands. |
| Medium | `_rollout_return` lines 156–166 adds a sender fact just before the send branch delivers. This can affect sender memory and later actions, particularly when reconstructed observation facts differ from the candidate. | Authorize the candidate without mutating task-visible sender memory or symmetrically initialize both clones, then compare branch states. |
| Medium | `_summarize` lines 708–714 stores a total under `mean_message_count`. Its H4 interval and point estimate differ in aggregation and zero handling (lines 828–863). | Use explicit units and names, matched estimands and null for undefined relative effect. |
| Medium | `select_communication` lines 156–184 can call `max()` on an empty agnostic message shortlist when a receiver exists. Main caller checks candidate non-emptiness and positive shortlist configuration. | Return no-send for empty shortlisted messages and test the public selector edge case. |
| Medium | `deliver` lines 288–299 validates/updates sender slots serially. With several active senders, an earlier delivery could make content available for a later sender in the same call. Primary runs use one sender. | Validate against frozen pre-delivery availability before applying updates for stretch multi-sender work. |
| Medium | Default additive `evaluate_pairs` at lines 195–212 uses the oracle horizon for decentralized-estimator diagnostics when no separate horizon is provided. E07/E12 decision/return claims can be mistaken for online H=5 results. | Make both horizons explicit in each artifact and label H=15 estimator runs as exploratory. |

## 11. Missing experiments

- Prospective matched-budget whole-episode evaluation and paired equivalence for Aware versus Full at a meaningful margin.
- Representative held-out delayed contexts with full candidate IDs, actual online policy decisions, independently seeded high-precision oracle and repeated comparisons.
- New seed families isolated from tuning. Start with a power or precision analysis over pilot seed variability and track effective delayed-eligible seed count.
- Equal-size candidate/pair computation ablation, with role-only compared under the same episode rather than context-continuation protocol.
- Receiver posterior calibration with known latent positions, hazard age/duration and receiver memory. Particle diversity is insufficient.
- Actual stale-message ablation varying age while holding the message and receiver context matched. Include late trajectory contexts and hazards with varying persistence.
- Full selection latency and scaling across rollout budget, particles, shortlist and independent layouts. Receiver-dependent cost and multi-sender are stretch arms. Learned surrogate and TarMAC/CMV add cost with lower priority than completing the primary tests.

## 12. Prioritized improvement plan

| Priority | Scientific value | Likely compute | Work |
| --- | --- | --- | --- |
| 0 | Makes every primary result auditable | Low | Freeze estimands and code, log timestep/event and candidate/action/rollout rows, fix ablation and tie semantics, add exact branch and leakage tests. |
| 1 | Resolves main research claims | Moderate for H1/H3, high for full episodes | Use held-out seeds and stratified early, late and stale contexts. Predeclare delayed selection, cost handling and precision. Run aware/agnostic/role-only/Full under assigned budgets and report paired return. |
| 2 | Separates belief quality and search noise | Moderate | Vary particles and rollouts independently, equalize candidate count, audit calibration, recall and regret, and measure end-to-end decision latency. Include initially null and stale conditions. |
| 3 | Tests external validity and scalability | High | Add a second layout family, then multi-sender interaction and larger heterogeneous teams. Consider a surrogate only if measured latency justifies it. |

No expensive confirmatory sweep ran during this audit. The existing main default CLI is `python -m voc_sar --config configs/default.yaml`, and running it currently writes a timestamped directory to `results`. For a future reproducibility replicate, use an unchanged copy of the saved configuration except for setting `evaluation.output_dir` to `audit_replay/default`, and run `python -m voc_sar --config audit_replay/default.yaml`. The comparable archived run took about 23 minutes with six workers. The prospective experiment requires a budget controller and trace schema first. A useful cost envelope for planning is about 23 minutes per existing ten-seed ten-condition suite on comparable hardware, with actual runtime depending on eligible candidates, early termination, workers and budget design.

## 13. Recommended plots and tables

1. One result table per configuration with seeds, episodes per seed per condition, delayed-context count, intervals and pass/fail criterion. Include separate descriptive and confirmatory columns.
2. H1 per-seed denominator and fraction with seed-cluster interval and 30% threshold. Plot both oracle receiver values for each paired message and mark high oracle Monte Carlo uncertainty.
3. Full-episode return versus assigned budget, with realized messages and paired seed uncertainty. Show role-only and Full under the same policy and environment settings.
4. H3 agreement split into delayed and other contexts and send/no-send versus exact pair, with raw numerators. Include the paired gain and 0.10 reference line.
5. H4 paired per-seed send rates connected across costs, with message/opportunity counts, task return and rescues.
6. Bias and regret versus model/sampling budget with per-context and end-to-end runtime. Display useful-message and high-value filter denominators and a small failure table keyed to complete traces.

## 14. Overall assessment

This is an executed simulator study with an inspectable local-policy pipeline and useful exploratory ablations. It establishes cost-related selectivity for the tested policy and shows receiver differences in some contexts. The evidence does not establish the full methodological claim of a weighted receiver-belief advantage, a task-return advantage at matched communication budgets, or a reliable delayed-value improvement. The strongest next result would come from a genuinely receiver-blind comparator under prospectively matched budgets and an independently sampled delayed-value evaluation with auditable action traces.
