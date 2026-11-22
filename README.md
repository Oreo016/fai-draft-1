# Receiver-Aware Value of Communication for SAR

This package implements the core method in [`final.pdf`](./final.pdf) and the
experimental context in
[`receiver_aware_voc_professional (2) (1).pdf`](./receiver_aware_voc_professional%20(2)%20(1).pdf).
It is a controlled, empirical SAR-Grid simulator for studying whether an agent
should send an existing structured fact, to which teammate, and at what
software-level cost. It does not generate messages or model a radio/network.

## Run

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e .
python -m voc_sar --config configs/default.yaml --validate-only
python -m voc_sar --config configs/default.yaml
```

Each run creates a unique timestamped subfolder under `evaluation.output_dir`.
It stores per-step episode traces in `episodes.json`, held-out decisions and
paired rollout returns in `contexts.json`, and aggregated results in
`summary.json`. It also saves the exact config, source hashes, run status and a
gallery of SVG charts. The command line reports episode and
diagnostic progress as the work completes. It also saves one
`search-operations-<method>.gif` for each communication method, using that
method's first episode to show agent positions, victims, hazards, task actions,
and any communication sent. Valuation runs at every eligible episode decision
by default; `evaluation.max_valuation_contexts_per_episode` can set an explicit
positive decision cap for a separate capped run. Held-out contexts are used
only for offline diagnostics. The default samples prespecified steps across
each held-out trajectory and reports steps lost to early termination. H2 uses a separate prospective arm with the same
assigned message budget and eligibility rule for aware and agnostic policies,
alongside a return equivalence test against Full. The assigned arm forces a
send at the first eligible steps up to the quota. Send times can diverge when
the trajectories diverge. The summary records unused slots and retains all
assigned pairs for policy comparisons. A matched-count confidence interval is
reported only if all pairs spend the budget.
H3 uses delayed contexts, with eligibility defined by the oracle's H=15 versus
H=1 value difference exceeding the configured margin. All-context agreement is
reported separately.
Independent condition/seed episodes run in parallel using
`evaluation.parallel_workers` (default `6`). Set it to `1` to run sequentially.
The current default uses 10 fresh seeds, 64 particles, 32 paired online rollouts,
`Hmyo=1`, `HNM=5`, `Heval=15`, and λ in {0, 0.5}. The oracle uses 128 paired
rollouts with separate RNG seeds. The primary assigned budget is two messages.
Eligible messages are located or stabilized victim reports and currently
observed active hazards. An episode can send each fact kind, location and status
only once within this arm. The Aware–Full return equivalence margin is 0.25
discounted reward units. The report also presents mean and total rescues and
paired rescue effects for Aware versus Full and for the assigned-budget arm.
Archived runs and the audit report describe earlier configurations and
algorithms and need a fresh evaluation for comparison with this version.

## Implemented pipeline

- **Environment:** seeded layouts, local observation/history, Scout discovery,
  Medic stabilization and hazard disable, Carrier transport to the evacuation
  cell, dynamic hazards,
  shared event reward, and explicit pre-action message delivery.
- **Task and message policies:** deterministic role policies plus Never,
  Random, Full, Fixed-frequency, receiver-aware, and receiver-agnostic
  communication conditions. Only the Scout is an active sender in the primary
  controlled comparison. The sender tracks facts it has delivered to each
  receiver so the candidate filter can down-rank redundant reports.
- **Valuation:** a seeded particle approximation with model propagation,
  weighted resampling, local evidence conditioning and proposal refresh.
  Aware valuation shortlists actual message–receiver pairs. Agnostic valuation
  shortlists messages and uses a fixed synthetic Carrier/Medic mixture with
  sender-only features. Both use paired rollouts and matched transition
  randomness and subtract communication cost.
- **Oracle and analysis:** privileged counterfactual evaluation is isolated in
  the offline diagnostics. The runner reports episode returns, message volume,
  selection regret, estimator MAE, filter recall, useful-message rate,
  myopic/nonmyopic agreement, paired-rollout standard errors, bootstrap
  intervals, failure cases, runtime, and H1-H4 target comparisons.

See [`docs/PROJECT_PLAN.md`](./docs/PROJECT_PLAN.md) for the architecture,
explicit modeling choices, interpretation limits, and possible extensions.

## Important modeling choices

The brief does not define a complete low-level action grammar or belief
initialization procedure. To make the experiment executable without changing
its central question, this implementation uses `scan`, `stabilize`, `rescue`,
and `disable` task actions, Manhattan-adjacent interactions, and deterministic
fixed role policies. With `environment.evacuation_point` configured, the Carrier
picks up a stabilized victim, travels to the cell, and uses `rescue` again to
deliver. Legacy configs without that field use adjacent rescue. The filter
propagates fixed task policies between observed states and uses a soft
likelihood built from sender position, visible teammates and current facts.
Its receiver histories and observation likelihood are model approximations.

`Observation` contains no global simulator state. Environment diagnostic
properties and `CounterfactualOracle` are privileged and must not be used by a
decentralized policy. A receiver summary contains only messages the sender
knows it delivered; it is not a claim to know the receiver's complete private
history. Communication cost is excluded from task reward and subtracted once
from VoC, as specified by the papers.
