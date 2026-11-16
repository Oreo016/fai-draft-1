# Project design and research boundaries

The implementation follows the final brief in `../final.pdf`. The professional
proposal provides motivation and candidate extensions; it does not override the
final brief's parameter choices or core deliverables.

## End-to-end architecture

1. **SAR-Grid:** `SARGrid` owns positions, hidden victim/hazard state, random
   dynamics, and each agent's private observation history. The public decision
   cycle is `observe -> evaluate -> deliver -> act -> transition`. `deliver`
   returns updated local observations before task policies run.
2. **Task policies:** fixed Scout, Carrier, and Medic policies map one local
   `Observation` to one action. They do not receive an environment or a global
   state object.
3. **Communication:** the primary runner activates one sender, limits it to
   one directed message per step, and compares Never, Random, Full,
   Fixed-frequency, receiver-aware, and receiver-agnostic choices. Candidate
   messages are local victim/hazard facts plus the sender's current intention.
   The sender's receiver summary contains facts it knows it delivered, so
   redundant-message filtering does not assume access to all receiver memory.
4. **VoC:** a cheap shortlist uses role utility, freshness, and locally known
   teammate tracks. The 64-particle model samples worlds only from the
   sender's observation. Paired send/no-send continuations use common seeded
   hazard randomness; communication is disabled after the intervention.
5. **Oracle/evaluation:** the counterfactual oracle can clone the full
   simulator state, but it is called only from the offline held-out diagnostic
   path. JSON outputs preserve episode-level seed, condition, return, message,
   and failure data as well as aggregate confidence intervals and H1-H4
   comparisons. Candidate and selected-pair Monte Carlo standard errors are
   also retained for uncertainty analysis.

## Explicit assumptions where the papers are underspecified

- Actions are `up/down/left/right/stay`, `scan`, `stabilize`, `rescue`, and
  `disable`. Interactions work at Manhattan distance at most one. A carrier
  transports a stabilized victim by rescuing it; there is no separate carrying
  state.
- The Scout's `scan` discovers hidden victims and hazards in its square vision
  region. Other agents learn reported facts through their own observations or
  delivered messages. A fact's observation time is preserved for freshness.
- Hazards block movement while active, may move/toggle according to configured
  probabilities, and Medic disable has a configurable finite duration.
- Unknown map entities are sampled from a seeded generative prior. Sender
  facts and locally observed teammate tracks condition the particles; the
  receiver's inaccessible private history is not fabricated as known. The
  rollout initializes only from facts this sender knows it delivered and from
  the sampled receiver observation; this is an approximation to `ell_j`.
- Fixed role policies are deterministic and observation-Markov. This makes
  paired differences interpretable, but their performance is not evidence
  about learned or optimal decentralized policies.
- The task policy chooses the nearest locally known viable objective rather
  than maintaining a separate persistent assignment state `Q_t`. Add
  reservation/assignment dynamics only as a controlled extension, since they
  can change the counterfactual value being measured.
- The runner caps expensive estimator decisions per episode using
  `evaluation.held_out_contexts` divided across the configured seeds. It logs
  the number of contexts actually valued so that this compute cap is visible.

These are deliberately visible assumptions rather than hidden claims. They
are suitable for a first reproducible implementation and should be stress
tested before making claims about real-world SAR.

## Core metrics and hypotheses

- **H1:** fraction of held-out contexts with a receiver marginal-value
  difference greater than `receiver_value_difference`; compare with
  `receiver_differentiation_fraction`.
- **H2:** report awareness-vs-agnostic paired return difference and bootstrap
  interval, plus communication reduction relative to Full. The
  matched-return comparison uses the explicit `evaluation.return_match_tolerance`
  (default 1.0 discounted-return unit). The paired-return test includes only
  same-seed aware/agnostic episodes whose realized message counts differ by no
  more than `evaluation.message_budget_match_tolerance` (default 0) and both
  actually send at least one message; the common
  per-sender-step ceiling alone is not treated as a matched realized budget.
  A reduction alone is not considered an improvement if task return falls.
- **H3:** compare myopic and nonmyopic selected actions against the
  `Heval` oracle choice on the same held-out contexts.
- **H4:** compare the positive-net-VoC communication rate at the lowest and
  highest configured costs.

All confidence intervals use seeded bootstrap resampling. Episode-level
comparisons resample paired seeds; held-out decision metrics resample seed
clusters so contexts from one trajectory are not treated as independent.
Hypothesis targets are prespecified in `configs/default.yaml`; reported
outcomes are measurements, not guaranteed successes.

## Improvement opportunities (not silently folded into the core)

1. Calibrate the receiver belief against a held-out state-estimation benchmark
   and report coverage/Brier score; the current seeded prior is intentionally
   simple and should not be mistaken for a learned Bayesian filter.
2. Add confidence-aware selection (for example, require a lower confidence
   bound on VoC to exceed zero) and test the trade-off between message rate and
   false-positive communication.
3. Add richer environment realism only as controlled variants: line-of-sight
   sensing, role-specific vision, congestion/collisions, and explicit
   victim-degradation deadlines. Keep a static-grid ablation for causal
   comparability.
4. Extend one-sender counterfactuals to multiple simultaneous senders only
   after measuring interaction effects; pairwise additive VoC need not remain
   valid under message interference.
5. If compute becomes a bottleneck, fit a surrogate only on training contexts
   and compare its held-out calibration and regret against the paired
   estimator and oracle.
6. Add external CMV/TarMAC baselines only when their inputs, task policy,
   compute, and communication budgets can be matched; the current core uses
   deterministic non-learning baselines.

## Reproducibility

Use the same configured seed set and task policies across conditions. The
episode file retains per-seed records; the summary stores the complete config,
runtime, confidence intervals, diagnostics, and failures. The oracle is
diagnostic only and is never part of the executable decentralized selector.
