# Experiment Inventory and Integrity Audit

**Evaluation Date**: 2026-11-22  
**Analyst Role**: Skeptical Research Analyst  
**Project**: Receiver-Aware Value of Communication for Cooperative Multi-Agent Search-and-Rescue  
**Base Target Configuration**: Grid $10\times 10$, 3 victims, 2 hazards, horizon 100, $\gamma=0.9$, $p_{\mathrm{move}}=0.8$, $p_{\mathrm{toggle}}=0.2$, rewards: rescue 10, locate 1, stabilize 2, step penalty 0.05. Estimator: 64 particles, shortlist 8, 32 rollouts, $H_{\mathrm{myo}}=1, H_{\mathrm{NM}}=5, H_{\mathrm{eval}}=15$, unit cost 1.0, cost sweep $[0.0, 0.5]$, useful threshold $\epsilon=0$, oracle threshold $\tau=0.5$, 10 seeds (0--9), 100 held-out contexts, 1000 bootstrap samples.

---

## 1. Inventory of Result Directories and Files

### A. Root Standard Runs (`results/run_*`)
1. **`results/run_20261113_124225_572171`**
   - **Environment/Config**: Grid 10, $\gamma=0.99$, static hazards ($p_{\mathrm{move}}=0.0, p_{\mathrm{toggle}}=0.0$), $H=(1, 5, 15)$, 100 episodes, seeds: `[0..9]`. Capped valuation contexts.
   - **Files (11)**: `summary.json`, `episodes.json`, `config_used.json`, `graphs/` (10 SVGs, `index.html`, `README.txt`), 7 animated GIFs.
2. **`results/run_20261120_210403_230837`**
   - **Environment/Config**: Grid 10, $\gamma=0.99$, static hazards ($p=0, 0$), uncapped valuation contexts, 100 episodes, seeds: `[0..9]`.
   - **Files (11)**: `summary.json`, `episodes.json`, `config_used.json`, `graphs/`, 7 GIFs.
3. **`results/run_20261121_021746_338196`**
   - **Environment/Config**: Grid 30, $\gamma=0.99$, static hazards ($p=0, 0$), 200 episodes, seeds: `[0..19]`, $H_{\mathrm{eval}}=35$.
   - **Files (10)**: `summary.json`, `episodes.json`, `config_used.json`, `graphs/` (10 SVGs), 7 GIFs. (`overview.png` omitted).
4. **`results/run_20261121_061856_659878`**
   - **Environment/Config**: Grid 30, $\gamma=0.99$, static hazards ($p=0, 0$), 200 episodes, seeds: `[0..19]`, $H_{\mathrm{eval}}=35$.
   - **Files (11)**: `summary.json`, `episodes.json`, `config_used.json`, `graphs/`, 7 GIFs.
5. **`results/run_20261121_120515_271425`**
   - **Environment/Config**: Grid 10, $\gamma=0.99$, static hazards ($p=0, 0$), 100 episodes, seeds: `[0..9]`.
   - **Files (11)**: `summary.json`, `episodes.json`, `config_used.json`, `graphs/`, 7 GIFs.
6. **`results/run_20261121_123226_187116` (PRIMARY EVALUATION BASELINE)**
   - **Environment/Config**: Grid 10, $\gamma=0.90$, dynamic hazards ($p_{\mathrm{move}}=0.8, p_{\mathrm{toggle}}=0.2$), 100 episodes, seeds: `[0..9]`, $H=(1, 5, 15)$, 64 particles, 32 rollouts.
   - **Files (11)**: `summary.json`, `episodes.json`, `config_used.json`, `graphs/` (10 SVGs, index, readme), 7 GIFs.
7. **`results/run_20261121_161323_602973`**
   - **Environment/Config**: Grid 10, $\gamma=0.90$, dynamic hazards ($p=0.8, 0.2$), 100 episodes, seeds: `[0..9]`. Verification re-run.
   - **Files (11)**: `summary.json`, `episodes.json`, `config_used.json`, `graphs/`, 7 GIFs.
8. **`results/run_20261121_170936_173219`**
   - **Environment/Config**: Grid 10, $\gamma=0.90$, moderate hazard dynamics ($p_{\mathrm{move}}=0.2, p_{\mathrm{toggle}}=0.2$), 100 episodes, seeds: `[0..9]`.
   - **Files (11)**: `summary.json`, `episodes.json`, `config_used.json`, `graphs/`, 7 GIFs.
9. **`results/run_20261121_181949_360025`**
   - **Environment/Config**: Grid 10, $\gamma=0.80$, moderate hazard dynamics ($p_{\mathrm{move}}=0.2, p_{\mathrm{toggle}}=0.2$), 100 episodes, seeds: `[0..9]`.
   - **Files (11)**: `summary.json`, `episodes.json`, `config_used.json`, `graphs/`, 7 GIFs.

### B. Custom Parameter Sweep (`results/sweep_gamma_lambda`)
- **Files**: `results/sweep_gamma_lambda/summary.json`, `results/sweep_gamma_lambda/summary.csv`.
- **Content**: Grid sweep across $\gamma \in \{0.8, 0.9, 0.95, 0.99\}$ and $\lambda \in \{0.0, 0.1, 0.25, 0.5\}$.

### C. Research Studies E01--E12
Completed across two full batch executions:
- Batch 1: `results/experiments/all_20261121_153923_146992/`
- Batch 2: `results/experiments/all_20261121_182538_166981/`
Each experiment has dedicated run outputs in `results/experiments/E01`--`E12` and complete report artifacts in `reports/experiments/E01`--`E12`:
- **E01 (Message Conditioning)**: `results/experiments/E01/run_20261121_182611_236195/summary.json` (20 contexts, seeds 0--9).
- **E02 (Role-Only Baseline)**: `results/experiments/E02/run_20261121_182600_198353/summary.json` (20 contexts, seeds 0--9).
- **E03 (Particle Diagnostics)**: `results/experiments/E03/run_20261121_182612_760521/summary.json` ($N_p \in [16, 256]$, 20 contexts, seeds 0--9).
- **E04 (Noise & CRN)**: `results/experiments/E04/run_20261121_182739_910360/summary.json` ($N_{\mathrm{roll}} \in [8, 128]$, 20 contexts, seeds 0--9).
- **E05 (Continuation Bias)**: `results/experiments/E05/run_20261121_183115_027376/summary.json` (Aware vs Disabled vs Fixed continuation, 20 contexts, seeds 0--9).
- **E06 (Robust Selection & Null)**: `results/experiments/E06/run_20261121_182808_270757/summary.json` (2000 null messages, $\epsilon \in [0, 0.5]$, LCB rule, seeds 0--9).
- **E07 (Cost Sweep & Pareto)**: `results/experiments/E07/run_20261121_182638_142372/summary.json` ($\lambda \in [0, 1.0]$, 20 contexts, seeds 0--9).
- **E08 (Hypothesis Robustness)**: `results/experiments/E08/run_20261121_182652_564758/summary.json` (20 held-out test contexts, seeds 0--9, Holm adjustments).
- **E09 (Horizon Sweep)**: `results/experiments/E09/run_20261121_182735_053944/summary.json` ($H \in [1, 15]$, 20 contexts, seeds 0--9).
- **E10 (Staleness Validation)**: `results/experiments/E10/run_20261121_182649_059693/summary.json` (40 evaluated facts, seeds 0--9).
- **E11 (Environment Robustness)**: `results/experiments/E11/run_20261121_182723_737946/summary.json` (Grids 8--12, Dynamics low/high, Teams 2--3 agents).
- **E12 (Failures & Runtime)**: `results/experiments/E12/run_20261121_182749_519315/summary.json` (3 failure contexts, sub-component runtime profiling).

---

## 2. Integrity and Discrepancy Audit

1. **Seed Coverage**:
   - All standard 10x10 runs and research studies E01--E12 evaluated exactly 10 seeds: `seeds: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]`.
   - The two 30x30 runs evaluated 20 seeds: `seeds: [0..19]`.
   - No seeds are missing from the runs.

2. **NaNs and Empty Files**:
   - Zero files are empty (all JSON, CSV, and TEX files contain valid data).
   - In `results/run_*/summary.json`, fields for standard errors on non-evaluating baselines (e.g. `mean_selected_estimator_standard_error` for `never`, `full`, `fixed_frequency`) are serialized as `null`, which is structurally expected. No invalid numeric `NaN` or `Infinity` tokens exist.
   - In `results/run_20261121_021746_338196`, `overview.png` was omitted during generation, but the underlying JSON summaries and SVG graph files are fully intact.

3. **Baseline Reproducibility**:
   - Re-running the baseline under identical flags (`run_20261121_123226_187116` vs `run_20261121_161323_602973`) produced numerically identical results:
     - Return Aware: 3.436 vs 3.436
     - Comm Rate Aware: 0.167 vs 0.167
     - Differentiation Fraction: 0.2857 vs 0.2857
     - Agreement Gain: +0.040 vs +0.040
     - Cost rate drop: 0.860 vs 0.860
   - Studies E01--E12 across Batch 1 (`153923`) and Batch 2 (`182538`) reproduced all headline metrics to within floating point identity.

---

## 3. Tuning vs. Evaluation Seeds and Leakage Analysis

> [!WARNING]
> **Methodological Leakage Finding**: No disjoint seed split was established between parameter selection and hypothesis evaluation.

- **Identical Seeds**: All 12 research studies E01--E12 and the primary evaluation baseline `run_20261121_123226_187116` operated on the identical set of seeds `[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]`.
- **Threshold Selection**:
  - The communication useful threshold $\epsilon = 0.0$ and oracle value threshold $\tau = 0.50$ were hard-coded in `configs/default.yaml` and `configs/experiments/study_base.yaml`.
  - While E06 evaluated alternate thresholds $\epsilon \in \{0.1, 0.25, 0.5\}$ and an LCB rule on these seeds, the main hypothesis tests evaluated on the very same 10 seeds without a hold-out test set of seeds.
  - In E08, the analysis explicitly attempted to define "held-out contexts", but those contexts were generated within the same 10 trajectories (`seed + step`).
- **Impact**: True out-of-sample generalization error cannot be cleanly separated from within-sample trajectory fitting. Future work must strictly enforce e.g. Seeds 0--9 for tuning and Seeds 10--29 for evaluation.
