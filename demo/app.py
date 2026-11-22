"""Local web demo for the receiver-aware VoC SAR project."""

from __future__ import annotations

import json
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[1]
DEMO = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

import yaml
from voc_sar.communication.estimator import PairedRolloutEstimator, ParticleReceiverEstimator
from voc_sar.environment.sar_grid import SARGrid
from voc_sar.experiments.runner import _capture_search_frame, _run_episode
from voc_sar.policies.task import task_policies
from voc_sar.types import Role


def base_config() -> dict:
    with (ROOT / "configs" / "default.yaml").open(encoding="utf-8") as stream:
        return yaml.safe_load(stream)


def bounded_number(payload: dict, name: str, default: float, low: float, high: float) -> float:
    value = float(payload.get(name, default))
    if not low <= value <= high:
        raise ValueError(f"{name} must be between {low} and {high}")
    return value


def demo_config(payload: dict) -> tuple[dict, int, str, float, int]:
    config = base_config()
    seed = int(payload.get("seed", 42))
    if not 0 <= seed <= 1_000_000:
        raise ValueError("seed must be between 0 and 1000000")
    method = str(payload.get("method", "aware_nonmyopic"))
    if method not in {"aware_nonmyopic", "agnostic_nonmyopic", "full", "never", "random"}:
        raise ValueError("unknown communication policy")
    cost = bounded_number(payload, "cost", 0.0, 0.0, 1.0)
    move = bounded_number(payload, "hazard_move_probability", 0.2, 0.0, 1.0)
    horizon = int(payload.get("episode_horizon", 100))
    if not 10 <= horizon <= 100:
        raise ValueError("episode_horizon must be between 10 and 100")
    valuation_horizon = int(payload.get("valuation_horizon", 5))
    if valuation_horizon not in {1, 5}:
        raise ValueError("valuation_horizon must be 1 or 5")
    config["environment"]["horizon"] = horizon
    config["environment"]["hazard_move_probability"] = move
    scenario = str(payload.get("scenario", "research"))
    if scenario not in {"research", "quick_rescue"}:
        raise ValueError("unknown scenario preset")
    if scenario == "quick_rescue":
        config["environment"].update(
            grid_size=6, victims=1, hazards=0, horizon=horizon,
            hazard_move_probability=0.0, hazard_toggle_probability=0.0,
        )
    # Smaller compute settings keep a single interactive episode responsive.
    config["communication"]["particles"] = 32
    config["communication"]["rollout_count"] = 8
    config["communication"]["nonmyopic_horizon"] = valuation_horizon
    config["communication"]["myopic_horizon"] = 1
    config["evaluation"]["parallel_workers"] = 1
    return config, seed, method, cost, valuation_horizon


def estimate_decision(payload: dict) -> dict:
    config, seed, _method, cost, horizon = demo_config(payload)
    env_config = config["environment"]
    comm_config = config["communication"]
    env = SARGrid(env_config)
    observations = env.reset(seed)
    policies = task_policies(env.roles)
    sender_id = next((i for i, role in enumerate(env.roles) if role == Role.SCOUT), 0)
    receiver_role = str(payload.get("receiver", "medic"))
    if receiver_role not in {"carrier", "medic"}:
        raise ValueError("receiver must be carrier or medic")
    receiver_id = next(
        (i for i, role in enumerate(env.roles) if role.value == receiver_role), None
    )
    if receiver_id is None:
        raise ValueError(f"the selected teammate ({receiver_role}) is not in this scenario")
    estimator = PairedRolloutEstimator(
        ParticleReceiverEstimator(env_config, seed=seed), policies,
        particles=comm_config["particles"], rollout_count=comm_config["rollout_count"],
        gamma=float(env_config.get("gamma", 0.99)), cost_weight=cost,
        unit_cost=float(comm_config.get("unit_cost", 1.0)), seed=seed,
        record_rollouts=True,
    )
    for _ in range(int(env_config["horizon"])):
        observation = observations[sender_id]
        fact = next((item for item in observation.known_facts
                     if item.kind == "victim" and item.status in {"located", "stabilized"}), None)
        if fact is not None:
            estimate = estimator.estimate(sender_id, receiver_id, fact, horizon, observation)
            frame = []
            choice = None
            if estimate.value_of_communication > 0:
                from voc_sar.types import CommunicationChoice
                choice = CommunicationChoice(receiver_id, fact)
            _capture_search_frame(frame, env, [], choice, "fact comparison", seed)
            returns = list(estimate.paired_returns)
            no_send = sum(pair[1] for pair in returns) / len(returns) if returns else None
            send = sum(pair[0] for pair in returns) / len(returns) if returns else None
            return {
                "seed": seed, "step": env.step_count, "receiver": receiver_role,
                "fact": {"kind": fact.kind, "location": list(fact.location), "status": fact.status},
                "horizon": horizon, "cost_weight": cost,
                "frame": frame[0], "marginal_value": estimate.marginal_value,
                "cost": estimate.communication_cost, "net_value": estimate.value_of_communication,
                "standard_error": estimate.standard_error,
                "send_return": send, "no_send_return": no_send,
                "note": "Estimated from paired send/no-send rollouts from the same sampled situations.",
            }
        observations = env.deliver([None] * len(observations))
        actions = [policy.act(obs) for policy, obs in zip(policies, observations)]
        observations = env.step(actions).observations
        if env.terminated:
            break
    raise ValueError("The Scout did not discover a victim report in this scenario. Try another scenario number.")


def run_episode(payload: dict) -> dict:
    config, seed, method, cost, horizon = demo_config(payload)
    frames: list[dict] = []
    row = _run_episode(config, seed, method, cost, horizon, None, frames)
    return {
        "seed": seed, "method": method, "steps": row["steps"],
        "return": row["discounted_task_return"],
        "messages": row["messages"], "rescued": row["rescued_victims"],
        "frames": frames,
        "note": (
            "Quick-rescue preset: small 6×6 map, one victim, no hazards. One episode demonstrates mechanics; it is not research evidence."
            if payload.get("scenario") == "quick_rescue" else
            "One seeded simulator episode; outcome variation is not evidence for a hypothesis."
        ),
    }


class Handler(BaseHTTPRequestHandler):
    def send_json(self, value: dict, status: int = 200) -> None:
        body = json.dumps(value).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path in {"/", "/index.html"}:
            body = (DEMO / "index.html").read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        self.send_error(404)

    def do_POST(self) -> None:
        path = urlparse(self.path).path
        try:
            payload = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
            if not isinstance(payload, dict):
                raise ValueError("request body must be an object")
            if path == "/api/estimate":
                self.send_json(estimate_decision(payload))
            elif path == "/api/run":
                self.send_json(run_episode(payload))
            else:
                self.send_error(404)
        except ValueError as error:
            self.send_json({"error": str(error)}, 400)
        except Exception as error:
            self.send_json({"error": f"{type(error).__name__}: {error}"}, 500)

    def log_message(self, format: str, *args) -> None:
        print(f"[demo] {format % args}" if args else f"[demo] {format}")


def main() -> None:
    host, port = "127.0.0.1", 8765
    print(f"Information sharing demo: http://{host}:{port}")
    ThreadingHTTPServer((host, port), Handler).serve_forever()


if __name__ == "__main__":
    main()
