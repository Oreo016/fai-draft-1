# Combined Analysis: Your Experiments + Partner's Experiments

**Generated from**: skeptical read-only audit of
- Your results: `results/` (seeds 0–9, p_move=0.8, γ=0.9)
- Partner's results: `project/faiproj/results/` (seeds 610–659, p_move=0.3 for dedicated hypothesis runs)

> [!CAUTION]
> This document reports numbers read from stored result files only.
> Every claim is cited to its source path.
> Neither dataset alone is sufficient to draw strong conclusions; see §6 for limitations.

---

## 1. Experimental Inventory (Partner's Runs)

### 1.1 Standard Simulation Runs

| Run ID | γ | p_move | p_toggle | H1 fraction | H1 CI | H4 drop |
|--------|---|--------|----------|-------------|-------|---------|
| run_20261121_203742 | 0.9 | 0.3 | 0.2 | 0.3091 | [0.053, 0.559] | 0.838 |
| run_20261121_205024 | 0.9 | 0.0 | 0.0 | 0.3208 | [0.044, 0.579] | 0.864 |
| run_20261121_211526 | 0.99 | 0.3 | 0.2 | 0.3091 | [0.017, 0.596] | 0.824 |

Source: `project/faiproj/results/{run_id}/summary.json`

> [!NOTE]
> `run_20261121_205024` uses p_move=0.0, p_toggle=0.0 (static hazards — no hazard movement at all). This is an extreme ablation config, not the standard setting.
> `run_20261121_211526` uses γ=0.99, not 0.9 — a different discount factor than the pre-specified config.
> Both standard runs (`203742`, `211526`) use p_move=0.3 (slower hazards vs your p_move=0.8).

All three partner standard runs have small episode counts (seeds N/A in top-level config — exact count
not stored in top-level; episodes are determined by seeds × episodes_per_seed from the per-seed structure).
CIs are wide in all cases: H1 point estimates meet 0.30 target but lower confidence bounds fall
well below 0.30 in all three runs.

### 1.2 Dedicated H1 Evaluation Runs

| Run ID | Seeds | Contexts | Eligible | Differentiated | Fraction | CI (seed-cluster bootstrap) | Meets target |
|--------|-------|----------|----------|----------------|----------|-----------------------------|--------------|
| h1_evaluation_225928 | 610–659 (50) | 500 requested / 500 observed | 232 | 66 | 0.2845 | [0.148, 0.434] | False (both) |
| h1_evaluation_230305 | 610–659 (50) | 500 eligible requested / 838 observed | 500 | 156 | 0.3120 | [0.204, 0.426] | Point True; LCB False |

Source: `project/faiproj/results/h1_evaluation_*/summary.json`

**Key detail — two different eligibility protocols:**
- Run `225928`: eligibility = any context where both agents receive a message; 10 contexts per seed requested
  regardless of eligibility. 232 of 500 contexts qualified, meaning 54% of sampled contexts did not meet
  the eligibility criterion (`h1_eligible_contexts` = 0 for 16 of 50 seeds).
- Run `230305`: targeted sampling — kept drawing until 10 eligible contexts per seed were collected; 838 contexts
  observed to find 500 eligible ones. All 50 seeds met the target. This is the methodologically stronger run.

**Critical finding**: Even with 50 seeds and targeted sampling (run `230305`), the seed-cluster bootstrap CI
is [0.204, 0.426]. The lower bound (0.204) is BELOW the pre-specified threshold of 0.30.
`lower_confidence_bound_meets_target: False` in both runs.

### 1.3 H2 Fixed-Budget Runs (11 runs, 6 unique seed blocks)

Partner ran two series of H2 experiments testing different message schedules and budgets.
The definitive results are in the two consolidated JSON files.

**`h2_fixed_budget_consolidated.json`** (first-fifth-of-horizon schedule):
Source: `project/faiproj/results/h2_fixed_budget_consolidated.json`

| Budget | Seeds | Pairs | ΔJ (aware−agnostic) | 95% CI | CR vs Full | Supported |
|--------|-------|-------|---------------------|--------|------------|-----------|
| 5 (primary, pooled) | 310–459 (100) | 100 | +0.449 | [0.226, 0.715] | 0.939 | **Yes** |
| 10 (secondary, pooled) | 310–459 (100) | 100 | +0.541 | [0.303, 0.823] | 0.879 | **Yes** |
| 5 (schedule sensitivity) | 10–59 (50) | 50 | −0.379 | [−0.782, −0.038] | — | **Negative** |

> [!WARNING]
> The schedule-sensitivity run (evenly spaced over full horizon, seeds 10–59) gives ΔJ = −0.379
> with CI entirely below zero: aware does WORSE than agnostic under fixed-interval scheduling.
> This shows the advantage is schedule-dependent: aware selection helps only when agents can
> choose WHEN to send (value-based timing), not just WHAT to send.

**`h2_full_episode_budget_consolidated.json`** (whole-episode budget, any-time sends):
Source: `project/faiproj/results/h2_full_episode_budget_consolidated.json`

| Budget | Seeds | Pairs | ΔJ (aware−agnostic) | 95% CI | CR vs Full | Supported |
|--------|-------|-------|---------------------|--------|------------|-----------|
| 5 (primary, pooled 100 seeds) | 510–759 (100) | 100 | +0.511 | [0.288, 0.770] | 0.942 | **Yes** |
| 10 (secondary, pooled 100 seeds) | 510–759 (100) | 95† | +0.663 | [0.428, 0.937] | 0.887 | **Yes** |

†: 5 of 100 aware-method episodes failed to meet the quota (early termination) and are excluded.

### 1.4 H4 Cost Sensitivity Run

Source: `project/faiproj/results/h4_cost_sensitivity_20261122_005543_147362/summary.json`

- Seeds: 610–659 (50 seeds), p_move=0.3
- Positive-VoC rate at λ=0: **0.1516** (CI [0.125, 0.181])
- Positive-VoC rate at λ=0.5: **0.0287** (CI [0.024, 0.035])
- Absolute drop: 0.123; Relative drop (aggregate): **0.811**
- Paired relative rate reduction: mean **0.676**, CI [0.535, 0.782]
- `point_estimate_meets_target: True` (target ≥0.15 relative reduction)
- `paired_interval_supports_decrease: True`

---

## 2. Cross-Dataset Hypothesis Verdicts

### H1 — Receiver Differentiation

**Pre-specified threshold**: fraction of eligible matched contexts with |MV_j1 − MV_j2| > 0.5 ≥ 0.30.

| Dataset | Config | Seeds | Eligible | Fraction | CI | LCB ≥ 0.30 | Verdict |
|---------|--------|-------|----------|----------|----|------------|---------|
| Yours | p_move=0.8, γ=0.9, 10 seeds | 0–9 | 49 | 0.2857 | [0.143, 0.449] | No | Inconclusive |
| Partner run 225928 | p_move=0.3, γ=0.9, 50 seeds | 610–659 | 232 | 0.2845 | [0.148, 0.434] | No | Inconclusive |
| Partner run 230305 | p_move=0.3, γ=0.9, 50 seeds | 610–659 | 500 | 0.3120 | [0.204, 0.426] | No | Inconclusive |

**Combined verdict: H1 Inconclusive across all datasets.**

The point estimate in the larger partner run (0.312) marginally exceeds the threshold (0.30), but the
pre-specified decision rule requires the lower confidence bound to meet the target — it does not (0.204 < 0.30).
Slower hazards (p_move=0.3 vs 0.8) did not substantively change the fraction: 0.2845 vs 0.2857 in the
comparable runs. The increase to 0.312 in run `230305` is explained by targeted eligibility sampling
(only eligible contexts are counted), not by a true underlying increase in differentiation rate.

**Variance structure**: 16 of 50 seeds have zero eligible contexts (run 225928 protocol). This is a
structural problem: at p_move=0.3, many episodes still produce no contexts where both a Carrier and
Medic receive distinct messages with MV above τ. The seed-level heterogeneity is extreme —
per-seed fractions range from 0.0 (many seeds) to 1.0 (seeds 617, 622, 626, 639, 654, 657) — which
inflates the width of the seed-cluster CI.

**What slower hazards changed**: Under p_move=0.3, hazard messages now have MV > 0 in some contexts
(this was categorically false at p_move=0.8 where all hazard MV=0). However, the aggregate differentiation
fraction is unchanged, because the increase in hazard-message eligibility is offset by the decrease in the
fraction of those contexts where the difference exceeds 0.5 (hazard locations are still uncertain enough that
MV values cluster near zero or small positives, rarely creating a large spread between roles).

### H2 — Selectivity

**Pre-specified tests**: (1) communication reduction vs Full at matched return ≥ 0.20; (2) positive
aware−agnostic return difference at matched message budget with CI excluding zero.

| Dataset | Config | CR vs Full | ΔJ at matched budget | CI excl. 0? | Verdict |
|---------|--------|------------|---------------------|-------------|---------|
| Yours | p_move=0.8, 10 seeds | 0.857 | +0.572 (CI [0.040, 1.049]) | Yes | **Supported (CR test only)** |
| Partner H2 consolidated (budget=5, pooled) | p_move=0.3, 100 seeds | 0.939 | +0.449 (CI [0.226, 0.715]) | Yes | **Supported (both tests)** |
| Partner H2 consolidated (budget=10, pooled) | p_move=0.3, 100 seeds | 0.879 | +0.541 (CI [0.303, 0.823]) | Yes | **Supported (both tests)** |
| Partner full-episode budget=5 (pooled) | p_move=0.3, 100 seeds | 0.942 | +0.511 (CI [0.288, 0.770]) | Yes | **Supported (both tests)** |
| Partner full-episode budget=10 (pooled) | p_move=0.3, 100 seeds | 0.887 | +0.663 (CI [0.428, 0.937]) | Yes | **Supported (both tests)** |
| Partner schedule-sensitivity (evenly spaced) | p_move=0.3, 50 seeds | — | −0.379 (CI [−0.782, −0.038]) | No (negative) | **Not Supported** |

**Combined verdict: H2 Supported — but schedule-conditional.**

The positive aware-over-agnostic advantage replicates robustly across 100 seeds and two budget levels
under value-based timing. The schedule-sensitivity result is a critical negative finding: when sends are
forced at evenly spaced intervals (removing the agent's choice of WHEN to send), the aware policy performs
WORSE (ΔJ = −0.379, CI entirely negative). This means the selectivity advantage is not due to message
content selection alone — it also requires the freedom to time sends.

### H3 — Downstream Value of Nonmyopic Horizon

**Pre-specified threshold**: Agree(H_NM, oracle) − Agree(H_myo, oracle) ≥ 0.10.

| Dataset | Config | Nonmyopic agree | Myopic agree | Gain | CI | Meets target | Verdict |
|---------|--------|----------------|--------------|------|----|--------------|---------|
| Yours (p_move=0.8) | Main | 0.75 | 0.71 | +0.04 | not stored | No | **Not Supported** |
| Partner run 203742 (p_move=0.3) | — | 0.78 | 0.73 | +0.05 | [−0.23, 0.33] | No | **Not Supported** |
| Partner run 205024 (p_move=0.0) | — | 0.75 | 0.78 | −0.03 | [−0.23, 0.22] | No | **Not Supported** |
| Partner run 211526 (γ=0.99, p_move=0.3) | — | 0.78 | 0.73 | +0.05 | [−0.23, 0.33] | No | **Not Supported** |

**Combined verdict: H3 Not Supported across all datasets and all configs tested.**

The gain never exceeds +0.05 in any run, far below the 0.10 threshold. The CI in all partner runs
where it is stored is [−0.23, 0.33] — wide, symmetric around zero. The static-hazard ablation
(p_move=0.0) gives a *negative* gain (−0.03), suggesting nonmyopic look-ahead can slightly degrade
agreement under some conditions. At p_move=0.3, the gain is +0.05, same as your result at p_move=0.8:
slower hazards did not rescue H3.

### H4 — Cost Sensitivity

**Pre-specified threshold**: relative drop in positive-VoC rate from λ=0 to λ=0.5 ≥ 0.15.

| Dataset | Config | Seeds | Rate λ=0 | Rate λ=0.5 | Rel. drop | Paired CI | Meets target | Verdict |
|---------|--------|-------|----------|------------|-----------|-----------|--------------|---------|
| Yours | p_move=0.8, 10 seeds | 0–9 | 0.154 | 0.021 | 0.860 | — | Yes | **Supported** |
| Partner H4 run | p_move=0.3, 50 seeds | 610–659 | 0.1516 | 0.0287 | 0.811 (agg.) | [0.535, 0.782] | Yes | **Supported** |
| Partner run 203742 | p_move=0.3, ~10 seeds | — | 0.142 | 0.023 | 0.838 | [0.630, 0.874] | Yes | **Supported** |
| Partner run 205024 | p_move=0.0, ~10 seeds | — | 0.154 | 0.021 | 0.864 | [0.658, 0.853] | Yes | **Supported** |
| Partner run 211526 | γ=0.99, p_move=0.3 | — | 0.142 | 0.025 | 0.824 | [0.585, 0.858] | Yes | **Supported** |

**Combined verdict: H4 Supported — robust across all configurations.**

H4 is the most robustly supported hypothesis. Relative drops of 0.81–0.86 are consistent across
p_move=0.0, 0.3, 0.8, γ=0.9 and 0.99, and across 10 and 50 seeds. The drop far exceeds the 0.15 threshold
in all cases. However, this is not surprising: cost-penalized sending is mechanically guaranteed to reduce
the positive-VoC rate if any sends are cost-penalized out. The question of whether this drop reflects
meaningful behavioral change (vs trivial cost arithmetic) is not separately tested.

---

## 3. What Partner's Experiments Address vs What They Don't

### Limitations Addressed by Partner's Data

| Original Limitation | Addressed? | How |
|--------------------|------------|-----|
| Fast hazards (p_move=0.8): all hazard MV=0 | **Partially** | p_move=0.3 gives non-zero hazard MV, but H1 fraction unchanged (0.285) |
| Small N (10 seeds): underpowered | **Partially** | H1 uses 50 seeds; H2/H4 use 50–100 seeds; CIs still wide for H1 |
| H2 budget-matched test missing | **Yes** | Full fixed-budget protocol with 100 seed pairs; both budget levels tested |
| H2 schedule dependence unknown | **Yes** | Schedule-sensitivity run shows ΔJ = −0.379 under evenly spaced sends |
| H4 cost sensitivity under slower hazards | **Yes** | p_move=0.3 gives consistent result (drop=0.81) |

### Limitations NOT Addressed by Partner's Data

| Limitation | Status |
|-----------|--------|
| Oracle random stream shared with estimator | Still unaddressed; affects both datasets equally |
| Continuation bias (CRB) | Not re-tested under p_move=0.3 |
| H3 delayed-value definition | Remains under-specified; CI includes zero in all runs |
| Separate tuning seeds from evaluation seeds | Partner used seeds 610–659 for BOTH H1 and H4 (same seeds) |
| 31.1% null false-positive rate (E06) | Not re-calibrated under p_move=0.3 |
| Short holdout (100 contexts / 10 eps) in your data | Partner H2 uses 50–100 seeds, 50 ep/seed — better but still limited |

---

## 4. Config Differences Between Datasets

> [!IMPORTANT]
> The two datasets are NOT directly comparable episode-for-episode:
> - Your data: p_move=0.8 (fast hazards), seeds 0–9, 100 held-out contexts
> - Partner's main data: p_move=0.3 (slower hazards), seeds 610–659, 100–500 contexts
> - Partner run 211526: γ=0.99 (not 0.9) — this run's hypothesis values cannot be directly compared to others

The return scale appears to differ across runs (partner's pareto_frontier shows max return ~2.34
in `203742` and ~2.09 in `205024`; your best was ~3.44). This scale difference arises from:
- Different seeds produce different episode difficulty distributions
- p_move=0.3 vs 0.8 changes expected hazard MV, affecting communication value
- The myopic policy (which dominates the Pareto frontier in all runs) achieves 0 messages in ALL configs;
  this is a structural result — the myopic horizon is too short to identify value-positive sends reliably

---

## 5. The Pareto Frontier Anomaly

In ALL runs (both yours and partner's), the Pareto frontier is dominated by `aware_myopic` with **zero messages**:

| Run | Method at frontier | Messages | Return |
|-----|-------------------|----------|--------|
| Your baseline | aware_myopic | 0 | 3.436 |
| Partner 203742 | aware_myopic | 0 | 2.337 |
| Partner 205024 | aware_myopic | 0 | 2.088 |
| Partner 211526 | aware_myopic | 0 | 1.671 |

Sources: respective `summary.json` → `pareto_frontier` field.

> [!WARNING]
> The myopic policy with zero messages consistently dominates. This means the adaptive policy
> NEVER recommends sending under myopic horizon — a degenerate outcome. If the myopic horizon (H=1)
> is too short to see the value of any communication, the entire MV framework may be miscalibrated
> for this environment. This anomaly was present in your data and is reproduced in all partner runs.

---

## 6. Summary Verdict Table (Combined)

| Hypothesis | Pre-spec threshold | Your result | Partner result (best) | Joint verdict |
|-----------|-------------------|-------------|----------------------|---------------|
| H1 Receiver Differentiation | Fraction ≥ 0.30 (LCB) | 0.2857, LCB=0.143 | 0.312, LCB=0.204 | **Inconclusive** |
| H2 Selectivity (CR test) | CR ≥ 0.20 | 0.857 ✓ | 0.939 ✓ | **Supported** |
| H2 Selectivity (ΔJ test) | CI excl. 0 | +0.572, Yes ✓ | +0.511, Yes ✓ | **Supported** |
| H2 Schedule sensitivity | — | not tested | ΔJ=−0.379, CI entirely negative | **Negative finding** |
| H3 Downstream value | Gain ≥ 0.10 | +0.04 | +0.05 (best) | **Not Supported** |
| H4 Cost sensitivity | Drop ≥ 0.15 | 0.860 ✓ | 0.811 ✓ | **Supported (robust)** |

---

## 7. Three Most Serious Threats to Validity (Updated)

1. **H1 is structurally underpowered by episode design, not just sample size.**
   Even at 50 seeds × 10 eligible contexts = 500 contexts (run 230305), the LCB is 0.204, not 0.30.
   16% of seeds produce zero eligible contexts. The problem is eligibility sparsity (many episodes have no
   contexts where both a Carrier and Medic receive messages), not just N. Source: both H1 summary JSONs.

2. **H2 advantage is schedule-conditional — the fixed-interval result is significantly negative.**
   ΔJ = −0.379 (CI [−0.782, −0.038]) when sends are evenly spaced. This means receiver-aware selection
   adds value only when combined with value-based timing. The preregistered H2 test does not separate
   these two effects. Source: `h2_fixed_budget_consolidated.json` → `schedule_sensitivity`.

3. **H3 is never supported and shows near-zero gain across all configs (−0.03 to +0.05).**
   The CI [−0.23, 0.33] in all partner runs is symmetric around zero. No configuration — faster hazards,
   slower hazards, static hazards, higher gamma — moves the gain above 0.10. The oracle agreement metric
   may be insensitive to horizon quality if the oracle itself is noisy (SE ≈ 0.082 ≈ 42% of MAE).
   Source: partner `summary.json` files, all three standard runs.

---

*Analysis written to `analysis_final/notes/combined_analysis.md`.*
*All numbers cited from `project/faiproj/results/*/summary.json` and partner's consolidated JSONs.*
