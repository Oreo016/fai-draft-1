# Empirical Limitations and Validity Audit

This document details seven critical methodological and structural limitations identified during the comprehensive audit of the Receiver-Aware VoC framework experiments.

---

## 1. Hazard Dynamics and Temporal Information Decay (Fast Hazards)
- **Configuration**: Hazard move probability $p_{\mathrm{move}} = 0.8$, toggle probability $p_{\mathrm{toggle}} = 0.2$.
- **Empirical Findings**:
  - In E10 (`reports/experiments/E10/run_20261121_182649_059693/episodes.csv`), the sole evaluated hazard message had marginal value $\mathrm{MV} = 0.0$.
  - In contrast, victim location messages achieved values up to $\mathrm{MV} = 2.020$.
  - Because hazards relocate with 80% probability every time step, spatial hazard warnings become obsolete within 1--2 steps.
- **Threat to Validity**:
  - Hypothesis H3 (downstream horizon value) is carried entirely by static victim messages.
  - Spatial coordination around dynamic obstacles provides no measurable downstream benefit under $p_{\mathrm{move}}=0.8$. The evaluation environment overpenalized hazard communication.

---

## 2. Short Evaluation Horizons and Incomplete Task Chains
- **Configuration**: Myopic $H_{\mathrm{myo}}=1$, Nonmyopic $H_{\mathrm{NM}}=5$, Oracle $H_{\mathrm{eval}}=15$.
- **Empirical Findings**:
  - At $H=1$, the myopic policy sent **0.00 messages** across the entire 100-step episode (`results/run_20261121_123226_187116/summary.json`). Because navigation to a victim requires multiple steps on a $10\times 10$ grid, a 1-step horizon cannot observe task completion rewards.
  - At $H=5$, rollouts observe victim discovery and stabilization rewards (rewards: 1.0 and 2.0), but rarely witness full rescue completion (reward: 10.0), which requires multi-step carrier transport.
  - Component-level reward breakdowns (locate, stabilize, rescue) were **not stored per component** in rollout trajectories.
- **Threat to Validity**:
  - The comparison between $H=1$ and $H=5$ is artificial: $H=1$ is completely communication-blind rather than sub-optimally communication-selective.
  - An evaluation horizon of $H=15$ on a $10\times 10$ grid truncates long-range carrier-medic cooperative rescue plans.

---

## 3. Oracle Ground-Truth Noise and Correlated Random Streams
- **Configuration**: Oracle rollout count $N_{\mathrm{roll}} = 32$, gamma $\gamma=0.90$.
- **Empirical Findings**:
  - In E04 (`results/experiments/E04/run_20261121_182739_910360/summary.json`), the standard error of the 32-rollout oracle estimate with CRN is $\mathrm{SE} = 0.0817$. Without CRN, it is $\mathrm{SE} = 0.1119$.
  - The baseline nonmyopic VoC MAE is $\mathrm{MAE} = 0.1961$ (`results/run_20261121_123226_187116/summary.json`). In E03, MAE ranges from $0.5817$ to $0.7116$.
  - The oracle standard error ($pprox 0.082$) constitutes **41.7% of the total reported MAE** ($0.1961$).
  - In `src/voc_sar/evaluation/oracle.py` and `src/voc_sar/communication/estimator.py`, both modules generate dynamics seeds via the formula:
    $$\mathrm{seed} + \mathrm{step} \times 65\,537 + \mathrm{rollout\_index}$$
    When initialized with identical seeds, the oracle and online estimator share dynamics trajectories rather than maintaining statistically independent evaluation streams.
- **Threat to Validity**:
  - Reported MAE figures are substantially corrupted by Monte Carlo variance in the oracle ground truth itself.
  - Shared random streams violate the independence assumption of offline counterfactual evaluation.

---

## 4. Estimator Noise and Null Hypothesis Calibration
- **Configuration**: Useful message threshold $\epsilon = 0.0$, oracle value threshold $\tau = 0.50$.
- **Empirical Findings**:
  - In E06 (`results/experiments/E06/run_20261121_182808_270757/summary.json`), evaluating the strict decision rule $\mathrm{VoC} > 0$ on 2,000 uninformative/redundant messages produced a **false-positive communication rate of 31.1%** ($0.311$).
  - Introducing an uncertainty threshold $\epsilon=0.25$ reduced false positives to $5.75\%$; the Lower Confidence Bound (LCB) rule reduced false positives to **0.30%** ($0.003$).
- **Threat to Validity**:
  - In the baseline configuration ($\epsilon=0.0$), nearly one-third of all transmitted messages represent noise-induced flukes rather than genuine task coordination.
  - At $\lambda=0.0$, this noise directly degrades task return ($J_0=1.658$ in E07), causing Aware to lose to zero-communication baselines.

---

## 5. Cost Scaling and Communication Quenching
- **Configuration**: Cost weight $\lambda \in [0.0, 0.5]$, unit cost $c_0 = 1.0$.
- **Empirical Findings**:
  - In `results/run_20261121_123226_187116/summary.json`, the 49 matched receiver value differences have median $0.00$, 75th percentile $0.5495$, and mean $0.3219$.
  - Exactly **71.43% of candidate message evaluations have marginal value $\mathrm{MV} < 0.50$**.
  - Setting $\lambda = 0.50$ imposes a penalty of $\lambda \cdot c_0 = 0.50$, automatically quenching over 71% of all messages.
- **Threat to Validity**:
  - The massive communication reduction observed under H4 (86.0% drop) is largely a threshold-clipping artifact of the chosen cost scale rather than continuous smooth cost-aware sensitivity.

---

## 6. Belief Filtering and Rejection Sampling Invariance
- **Configuration**: 64 particles, shortlist size 8.
- **Empirical Findings**:
  - In E01 (`results/experiments/E01/run_20261121_182611_236195/summary.json`), conditioning belief particles on message truth produced **identically zero change** in MAE ($0.000$), regret ($0.000$), and task return ($0.000$).
  - In E03 (`results/experiments/E03/run_20261121_182612_760521/summary.json`), effective sample size (ESS) fraction was $1.000$ and particle degeneracy was $0.000$ across all particle counts $N_p \in [16, 256]$.
- **Threat to Validity**:
  - The belief update filter acts as an unweighted rejection sampler without sequential particle depletion.
  - The lack of empirical sensitivity in E01 indicates that the current observation model does not stress the receiver-aware belief conditioning mechanism.

---

## 7. Sample Size, Seed Reuse, and Multiple Testing Penalties
- **Configuration**: 10 seeds (`[0..9]`), 4 hypotheses (H1--H4).
- **Empirical Findings**:
  - With only 10 seeds, statistical tests suffer severe power penalties.
  - In E08 (`results/experiments/E08/run_20261121_182652_564758/summary.json`), H4 achieved raw $p = 0.0671$, but after Holm adjustment across H1--H4, the adjusted $p$-value rose to **$0.2013$**, rendering the result formally non-significant at $\alpha=0.05$.
  - H2 and H3 suffered Holm-adjusted $p$-values of **$1.0000$**.
  - All studies (E01--E12) reused the exact same 10 seeds without an isolated out-of-sample test cohort.
- **Threat to Validity**:
  - The 10-seed evaluation cohort is statistically underpowered for Dec-POMDP rollouts with high environment variance.
  - Lack of separate tuning and evaluation seed sets creates vulnerability to within-sample overfitting.
