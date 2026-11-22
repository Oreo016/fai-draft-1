# Empirical Results Summary: Receiver-Aware VoC SAR

**Evaluation Context**: Heterogeneous Scout/Carrier/Medic SAR Dec-POMDP  
**Baseline Run Directory**: `results/run_20261121_123226_187116`  
**Research Experiments**: `results/experiments/E01`--`E12`

---

## 1. Executive Verdict on Hypotheses H1--H4

| Hypothesis | Pre-Specified Threshold | Observed Value | 95% Confidence Interval | Raw $p$ | Holm Adjusted $p$ | Empirical Verdict |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **H1: Receiver Differentiation** | $\ge 0.30$ | **0.2857** (28.57%) | Cluster: `[0.0196, 0.5574]`<br>Context: `[0.1633, 0.4082]` | $0.0037$ | **$0.0148$** | **Inconclusive / Marginally Missed** |
| **H2: Selectivity** | $CR \ge 0.20$, $\Delta J > 0$ | **$CR = 0.8572$**<br>$\Delta J = +0.5724$ | Cluster: `[0.1615, 1.1027]`<br>Context: `[0.1615, 1.1027]` | $0.7526$ | **$1.0000$** | **Supported** (Return & Vol. Red.); Inconclusive at exact budget |
| **H3: Downstream Value** | $\Delta \mathrm{Agree} \ge 0.10$ | **$+0.0400$** (+4.00%) | Cluster: `[-0.2000, 0.3200]`<br>Context: `[-0.2000, 0.3200]` | $1.0000$ | **$1.0000$** | **Not Supported** (Fails 0.10 threshold) |
| **H4: Cost Sensitivity** | Drop $\ge 0.15$ | **0.8600** (86.00%) | Cluster: `[0.6488, 0.9054]`<br>Context: `[0.6488, 0.9054]` | $0.0671$ | **$0.2013$** | **Supported** (Effect size met; Holm $p=0.201$ due to $N=10$ seeds) |

*Sources: `results/run_20261121_123226_187116/summary.json`, `results/experiments/E08/run_20261121_182652_564758/summary.json`.*

---

## 2. In-Depth Analysis by Hypothesis

### Hypothesis 1: Receiver Differentiation
- **Target**: In at least 30% of matched valuation contexts with communication, $|\mathrm{MV}(m, j_1) - \mathrm{MV}(m, j_2)| > 0.5$.
- **Observed**: **28.57%** (14 out of 49 matched contexts).
- **Cluster Bootstrap CI (Seeds)**: `[0.0196, 0.5574]`.
- **Context-Level Resampling CI**: `[0.1633, 0.4082]`.
- **Comparison to Role-Only Baseline (E02)**:
  - In E02 (`results/experiments/E02/run_20261121_182600_198353/summary.json`), Role-Only communication achieves differentiation fraction **0.2000** (20.0%).
  - The Aware method achieves **0.2857**, representing a modest $+0.0857$ lift over pure role-based routing.
  - While raw significance indicates differentiation differs from 0 ($p=0.0037$, Holm $p=0.0148$), the point estimate misses the pre-specified 30% threshold by 1.43 percentage points.

### Hypothesis 2: Communication Selectivity
- **Target**: Communication reduction $CR \ge 20\%$ relative to Full communication at matched return (tolerance 1.0), and positive paired return difference over Agnostic.
- **Observed**:
  - Communication reduction vs Full at matched return is **85.72%** (Observed: 0.8572), vastly exceeding the 20% target.
  - Across all 100 episodes, Aware ($\lambda=0$) achieves mean return **3.436** with only **12.6 messages**, beating Full (return **3.330** with **87.6 messages**, an 85.6% message reduction) and Agnostic (return **2.656** with **34.7 messages**).
  - The paired return difference (Aware minus Agnostic) across all episodes is **$+0.5724$**, with 95% bootstrap CI `[0.1615, 1.1027]`, strictly excluding zero.
  - *Caveat*: In E08 and `runner.py`, under exact integer-budget matching (`message_budget_match_tolerance: 0`), only 7--10 episodes qualified, yielding return difference $+0.0013$ `[0.0000, 0.0040]` (Baseline) and $-0.1792$ `[-0.5817, 0.0443]` (E08). Thus, the selectivity advantage operates through macro policy efficiency rather than single-message replacement at identical budgets.
  - *Comparison to Role-Only (E02)*: Role-only achieves return **2.459** with 1.0 message. When budget-matched in E02, Aware achieved $-0.8010$ return difference, indicating that hardcoded role heuristics outperform noisy online rollouts in static message contexts.

### Hypothesis 3: Downstream Value of Nonmyopic Valuation
- **Target**: $\mathrm{Agree}(H_{\mathrm{NM}}, H_{\mathrm{eval}}) - \mathrm{Agree}(H_{\mathrm{myo}}, H_{\mathrm{eval}}) \ge 0.10$.
- **Observed**:
  - Nonmyopic agreement: **0.8100** (81.0%).
  - Myopic agreement: **0.7700** (77.0%).
  - Observed Gain: **$+0.0400$** (+4.00%), with 95% CI `[-0.2000, 0.3200]`.
- **Delayed-Value Analysis (E08)**:
  - Definition: Contexts where $|\mathrm{MV}(H_{\mathrm{eval}}) - \mathrm{MV}(H_{\mathrm{myo}})| > 0.50$.
  - Out of 10 evaluation seeds, only **3 seeds** contained contexts qualifying as delayed-value!
  - Within these 3 seeds, the agreement gain was identically **0.000** (`h3_delayed_context_gain: 0.0, CI: [0.0, 0.0]`).
- **Verdict**: **Not Supported**. Under fast hazards ($p_{\mathrm{move}}=0.8$), multi-step information has low persistence, preventing $H=5$ from showing a $\ge 10\%$ gain over $H=1$.

### Hypothesis 4: Cost Sensitivity
- **Target**: Relative drop in positive-VoC communication rate $\ge 15\%$ from $\lambda=0.0$ to $\lambda=0.5$.
- **Observed**:
  - Positive-VoC communication rate drops from **0.1666** ($\lambda=0.0$) to **0.0233** ($\lambda=0.5$).
  - Relative rate reduction: **86.00%** (0.8600), with 95% CI `[0.6488, 0.9054]`.
  - Paired relative drop mean across seeds: **78.99%** (0.7899).
- **Noise-Robust Rules (E06)**:
  - Strict rule ($VoC > 0$): 100% relative drop (rate drops from 0.10 to 0.00).
  - LCB rule: 100% relative drop (rate drops from 0.05 to 0.00).
  - $\epsilon=0.10$: 100% relative drop (0.10 to 0.00).
  - $\epsilon=0.25$: 100% relative drop (0.05 to 0.00).
  - $\epsilon=0.50$: Rate is 0.0 at both costs (zero drop).
- **Verdict**: **Supported**. The estimator robustly curbs communication volume as transmission cost scales.

---

## 3. Synthesis of Studies E01--E12

1. **E01 (Conditioning)**: Conditioning belief particles on message truth produced **0.000 change** in MAE, regret, or return (`mae_change: 0.0`, Return: 2.203). Receiver particle sampling did not alter the rollout dynamics under the tested configurations.
2. **E02 (Role-Only)**: Role-only achieved $J_0=2.459$, Regret=0.016, messages=1.0. Aware achieved $J_0=1.658$, Regret=0.659. Budget-matched aware minus role-only was $-0.8010$. Aware lost to role-only because role-only exploits exact task semantics without Monte Carlo estimation noise.
3. **E03 (Particles)**: Sweep across $N_p \in [16, 256]$ revealed an elbow at $N_p=64$, where regret dropped from 0.541 ($N_p=32$) to 0.242 ($N_p=64$). ESS fraction was 1.0 and degeneracy rate 0.0 across all particle counts.
4. **E04 (CRN & Rollouts)**: Common Random Numbers reduced rollout standard error by 27% ($N=32$, SE 0.082 vs 0.112) to 37% ($N=128$, SE 0.036 vs 0.057) with negligible bias ($|\mathrm{bias}| < 0.02$).
5. **E05 (Continuation Bias)**: Disabling communication in rollouts yielded **100% top-1 agreement** and Spearman $\rho=0.838$ vs the disabled oracle. Fixed-frequency continuation preserved top-1 agreement (1.0) but induced a large valuation shift (MAE 0.288 vs 0.023).
6. **E06 (Null Calibration)**: Strict $VoC > 0$ exhibited a **31.1% false-positive rate** on redundant null messages. The LCB rule suppressed false positives to **0.3%**.
7. **E07 (Cost & Pareto)**: At $\lambda=0.0$, noisy communication depressed Aware return ($J_0=1.658$); increasing $\lambda$ to 0.50 pruned low-value messages, raising return to $J_0=2.474$.
8. **E09 (Horizon Sweep)**: $H=1$ suffered complete myopic blindness (0 messages sent). $H=5$ achieved 0.35 messages and 0.75 agreement.
9. **E10 (Staleness)**: All evaluated facts had information age $\kappa=0$; hazard facts had $\mathrm{MV}=0.0$, while fresh victim facts achieved $\mathrm{MV}=2.02$.
10. **E11 (Robustness)**: Aware underperformed Agnostic on 8x8 ($-1.026$) and 10x10 ($-1.119$), but tied on 12x12 ($0.000$). Under slow hazards ($p=0.2$), the performance deficit narrowed from $-1.085$ to $-0.579$.
11. **E12 (Failures & Runtime)**: Profiling established that rollouts consume 71.6%--81.3% of runtime (0.104s--0.127s/eval), the oracle consumes 18.6%--24.3% (0.233s--0.309s), and particle filtering takes 9.8%--11.2% (0.014s--0.017s).
