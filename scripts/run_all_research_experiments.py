"""Run all E01-E12 studies concurrently and persist a progress tracker."""

from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import time

from voc_sar.research_experiments.common import load_study_config


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_IDS = [f"E{index:02d}" for index in range(1, 13)]


def _write_progress(path: Path, state: dict) -> None:
    state["updated_at_utc"] = datetime.now(timezone.utc).isoformat()
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(state, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)


def _run_one(experiment_id: str, log_path: Path) -> dict:
    command = [sys.executable, str(ROOT / "scripts" / "run_research_experiment.py"), experiment_id]
    started = time.perf_counter()
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=False)
    return {
        "experiment_id": experiment_id,
        "return_code": process.returncode,
        "runtime_seconds": time.perf_counter() - started,
        "log_file": str(log_path.resolve()),
    }


def main() -> int:
    configured_workers = int(load_study_config("E01")["evaluation"].get("parallel_workers", 6))
    workers = max(1, min(configured_workers, len(EXPERIMENT_IDS)))
    stamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S_%f")
    run_root = ROOT / "results" / "experiments" / f"all_{stamp}"
    logs_dir = run_root / "logs"
    configs_dir = run_root / "configs"
    logs_dir.mkdir(parents=True, exist_ok=False)
    configs_dir.mkdir(parents=True, exist_ok=False)
    config_files = {}
    for experiment_id in EXPERIMENT_IDS:
        config_path = configs_dir / f"{experiment_id}_config.json"
        config_path.write_text(
            json.dumps(load_study_config(experiment_id), indent=2, sort_keys=True),
            encoding="utf-8",
        )
        config_files[experiment_id] = str(config_path.resolve())
    progress_path = run_root / "progress.json"
    state = {
        "status": "running",
        "workers": workers,
        "experiment_ids": EXPERIMENT_IDS,
        "completed_count": 0,
        "total_count": len(EXPERIMENT_IDS),
        "experiments": {
            experiment_id: {"status": "queued", "config_file": config_files[experiment_id]}
            for experiment_id in EXPERIMENT_IDS
        },
        "log_directory": str(logs_dir.resolve()),
    }
    _write_progress(progress_path, state)
    print(f"Running {len(EXPERIMENT_IDS)} studies with {workers} workers.", flush=True)
    print(f"Progress: {progress_path.resolve()}", flush=True)
    pending = {}
    with ThreadPoolExecutor(max_workers=workers) as executor:
        next_index = 0
        while next_index < min(workers, len(EXPERIMENT_IDS)):
            experiment_id = EXPERIMENT_IDS[next_index]
            state["experiments"][experiment_id]["status"] = "running"
            pending[executor.submit(_run_one, experiment_id, logs_dir / f"{experiment_id}.log")] = experiment_id
            next_index += 1
            _write_progress(progress_path, state)
        while pending:
            completed, _ = wait(pending, timeout=5, return_when=FIRST_COMPLETED)
            for future in completed:
                experiment_id = pending.pop(future)
                try:
                    result = future.result()
                except Exception as error:  # record launch failures and keep other jobs alive
                    result = {"experiment_id": experiment_id, "return_code": -1, "error": repr(error)}
                result["status"] = "complete" if result.get("return_code") == 0 else "failed"
                state["experiments"][experiment_id] = result
                state["completed_count"] += 1
                _write_progress(progress_path, state)
                print(f"Progress: {state['completed_count']}/{state['total_count']} complete; {experiment_id} {result['status']}.", flush=True)
                if next_index < len(EXPERIMENT_IDS):
                    queued_id = EXPERIMENT_IDS[next_index]
                    state["experiments"][queued_id]["status"] = "running"
                    pending[executor.submit(_run_one, queued_id, logs_dir / f"{queued_id}.log")] = queued_id
                    next_index += 1
                    _write_progress(progress_path, state)
    failures = [key for key, value in state["experiments"].items() if value["status"] != "complete"]
    state["status"] = "failed" if failures else "complete"
    state["failures"] = failures
    _write_progress(progress_path, state)
    print(f"All-study status: {state['status']}", flush=True)
    print(f"Progress file: {progress_path.resolve()}", flush=True)
    print(f"Logs: {logs_dir.resolve()}", flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
