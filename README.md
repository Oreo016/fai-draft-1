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
It stores `episodes.json`, `summary.json`, and a `graphs` folder with standalone
SVG charts plus an `index.html` gallery. It also saves one
`search-operations-<method>.gif` for each communication method, using that
method's first episode to show agent positions, victims, hazards, task actions,
and any communication sent. The default experiment uses the paper's 10 seeds,
64 receiver-state particles, 32 paired rollouts, `Hmyo=1`, `HNM=5`, `Heval=15`,
and cost sweep `λ in {0, 0.5}`. The offline diagnostics use a separate
held-out, no-communication trajectory set. Results are generated locally;
no performance claim is made by this source tree alone.

## Implemented pipeline

- **Environment:** seeded layouts, local observation/history, Scout discovery,
  Medic stabilization and hazard disable, Carrier rescue, dynamic hazards,
  shared event reward, and explicit pre-action message delivery.
- **Task and message policies:** deterministic role policies plus Never,
  Random, Full, Fixed-frequency, receiver-aware, and receiver-agnostic
  communication conditions. Only the Scout is an active sender in the primary
  controlled comparison. The sender tracks facts it has delivered to each
  receiver so the candidate filter can down-rank redundant reports.
- **Valuation:** a seeded 64-particle prior conditioned on the sender's local
  evidence, cheap role/freshness/proximity shortlist, paired send/no-send
  rollouts, matched transition randomness, and communication cost subtraction.
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
and `disable` task actions; Manhattan-adjacent interactions; a seeded,
sender-conditioned generative prior for unknown locations; and deterministic
fixed role policies. These choices are configurable or replaceable behind the
environment and policy interfaces. They are approximations, not claims about a
physical SAR system.

`Observation` contains no global simulator state. Environment diagnostic
properties and `CounterfactualOracle` are privileged and must not be used by a
decentralized policy. A receiver summary contains only messages the sender
knows it delivered; it is not a claim to know the receiver's complete private
history. Communication cost is excluded from task reward and subtracted once
from VoC, as specified by the papers.
