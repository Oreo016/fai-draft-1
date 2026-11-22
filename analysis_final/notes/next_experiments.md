# Prioritized Next Experiments: Resolving Core Empirical Weaknesses

This document presents a ranked list of the highest-yield, cheapest additional experiments required to resolve the validity threats identified during the analytical audit.

---

## Ranked Recommendations

### Rank 1: Slower Hazard Dynamics Sweep ($p_{\mathrm{move}} \in [0.05, 0.20]$)
- **Weakness Addressed**: Fast hazard movement ($p=0.8$) renders hazard communication immediately obsolete (E10 observed $\mathrm{MV}=0.0$), forcing Hypothesis H3 to be carried solely by victim messages.
- **Experimental Protocol**:
  - Evaluate $p_{\mathrm{move}} \in \{0.05, 0.10, 0.20\}$ and $p_{\mathrm{toggle}} = 0.05$ on the standard $10\times 10$ grid.
  - Track $\mathrm{MV}$ across non-zero information ages $\kappa \in \{1, 2, 4, 8\}$.
- **Expected Outcome**: Spatial hazard warnings maintain non-zero value over horizons $H=5$ and $H=15$, allowing true dynamic obstacle avoidance coordination to manifest.
- **Estimated Compute Cost**: **$1.0\times$** relative to baseline evaluation (approx. 15 minutes on 6 CPU workers).

---

### Rank 2: Independent, High-Rollout Oracle Stream ($N_{\mathrm{roll}}=128$, Independent Seeds)
- **Weakness Addressed**: Current oracle standard error ($pprox 0.082$) is 42% of reported MAE ($0.196$), and oracle/estimator share identical paired seeds (`self.seed + step * 65537 + rollout_index`).
- **Experimental Protocol**:
  - Decouple oracle RNG stream by adding an independent seed offset (`seed + 1_000_003`).
  - Increase oracle rollouts from $N_{\mathrm{roll}}=32$ to $N_{\mathrm{roll}}=128$.
- **Expected Outcome**: Oracle SE drops to $\le 0.036$, providing a genuine, noise-free ground truth against which estimator MAE and regret become rigorously interpretable.
- **Estimated Compute Cost**: **$1.5\times$** relative to baseline (oracle evaluation accounts for $\approx 20\%$ of runtime).

---

### Rank 3: Statistically Powered 30-Seed Evaluation with Disjoint Tuning/Evaluation Split
- **Weakness Addressed**: The 10-seed evaluation cohort is underpowered; Holm correction inflates $p$-values to non-significance (H4 Holm $p=0.2013$, H2/H3 Holm $p=1.0000$). Seeds 0--9 were reused across all tuning studies.
- **Experimental Protocol**:
  - Designate Seeds 0--9 as the Tuning Cohort (for setting $\epsilon, \lambda, H$).
  - Designate Seeds 10--29 (20 seeds) as the Strict Evaluation Cohort.
- **Expected Outcome**: Eliminates all risk of within-sample leakage; provides sufficient statistical power for paired sign-flip permutation tests under Holm-Bonferroni correction.
- **Estimated Compute Cost**: **$2.0\times$** relative to baseline (20 additional seeds).

---

### Rank 4: Extended Planning and Evaluation Horizons ($H_{\mathrm{NM}}=10, H_{\mathrm{eval}}=25$)
- **Weakness Addressed**: $H=1$ suffers total communication blindness (0 messages sent) because multi-step rescue cannot complete in 1 step; $H=5$ rarely observes carrier delivery (reward 10).
- **Experimental Protocol**:
  - Run horizon ladder: $H_{\mathrm{myo}}=2, H_{\mathrm{NM}}=8, H_{\mathrm{eval}}=25$.
  - Explicitly log reward decomposition per rollout (locate vs stabilize vs rescue).
- **Expected Outcome**: Allows multi-step carrier delivery chains to complete, enabling Hypothesis H3 to achieve its pre-specified $\ge 10\%$ agreement gain target.
- **Estimated Compute Cost**: **$2.5\times$** relative to baseline (rollout duration scales with horizon).

---

### Rank 5: Second Domain Instance ($15\times 15$ Grid, 5 Victims, 3 Hazards)
- **Weakness Addressed**: All primary conclusions currently rely on a single $10\times 10$ environment topology.
- **Experimental Protocol**:
  - Test scalability on a $15\times 15$ grid with 5 victims and 3 hazards over $T=150$ steps across 10 seeds.
- **Expected Outcome**: Proves that receiver selectivity gains scale with search area and victim density.
- **Estimated Compute Cost**: **$3.5\times$** relative to baseline.

---

## Summary of Relative Resource Costs

| Recommendation | Priority | Addressed Weakness | Relative Cost | Time (6 Workers) |
| :--- | :---: | :--- | :---: | :---: |
| **Slower Hazard Dynamics** | **High** | Eliminates immediate information obsolescence | $1.0\times$ | $\sim 15$ min |
| **Independent Oracle ($N=128$)** | **High** | Eliminates oracle noise and RNG stream confounding | $1.5\times$ | $\sim 22$ min |
| **30-Seed Split (Tuning vs Eval)** | **High** | Resolves Holm power loss and leakage threat | $2.0\times$ | $\sim 30$ min |
| **Longer Horizons ($H=8, 25$)** | **Medium** | Captures multi-step carrier rescue chains | $2.5\times$ | $\sim 40$ min |
| **$15\times 15$ Grid Instance** | **Medium** | Validates topological generalization | $3.5\times$ | $\sim 55$ min |
