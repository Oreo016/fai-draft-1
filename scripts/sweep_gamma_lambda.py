"""Cross-evaluation of VoC communication methods across discount factors (gamma) and cost weights (lambda)."""

import json
from pathlib import Path
from collections import defaultdict
from datetime import datetime
import random
import time
import yaml

from voc_sar.evaluation.metrics import bootstrap_mean_interval
from voc_sar.research_experiments.common import (
    collect_contexts, evaluate_pairs, load_study_config, mean, policy_choice, rollout_task_return
)
from voc_sar.research_experiments.e02_role_only import _compatibility


def run_sweep():
    started_at = time.perf_counter()
    base_config = load_study_config("E07")
    
    gammas = [0.80, 0.90, 0.95, 0.99]
    lambdas = [0.0, 0.05, 0.10, 0.25, 0.50, 1.00]
    methods = ["aware", "agnostic", "role_only", "fixed_frequency", "full", "never", "random"]
    
    compatibility = base_config.get("experiment", {}).get("compatibility", {})
    unit_cost_c0 = float(base_config["experiment"].get("unit_cost_c0", 1.0))
    bootstrap_samples = 2000
    confidence = 0.95
    
    results_by_gamma = {}
    csv_rows = []
    raw_rows = []
    
    print(f"Starting sweep across {len(gammas)} gammas and {len(lambdas)} lambdas...")

    for gamma in gammas:
        gamma_start = time.perf_counter()
        config = json.loads(json.dumps(base_config))
        config["environment"]["gamma"] = gamma
        config["communication"]["gamma"] = gamma
        
        # Collect decision contexts for this gamma
        contexts = collect_contexts(config)
        print(f"Gamma {gamma}: collected {len(contexts)} decision contexts across {len(config['evaluation']['seeds'])} seeds.")
        
        context_evaluations = []
        for context in contexts:
            messages = list(context["messages"][:3])
            truth, estimates = evaluate_pairs(config, context, messages=messages)
            context_evaluations.append({
                "context": context,
                "messages": messages,
                "truth": truth,
                "estimates": estimates,
            })
            
        records_by_lambda = {}
        pareto_points = []
        
        for lam in lambdas:
            method_rows = {m: [] for m in methods}
            
            for item in context_evaluations:
                context = item["context"]
                messages = item["messages"]
                truth = item["truth"]
                estimates = item["estimates"]
                
                receiver_ids = sorted({receiver for _, receiver in estimates})
                
                aware_key = policy_choice(config, context, messages, aware=True, cost_weight=lam)
                agnostic_key = policy_choice(config, context, messages, aware=False, cost_weight=lam)

                # 3. Role-only
                role_rank = [
                    (_compatibility(msg, context["roles"][rec], compatibility), -idx, -rec, idx, rec)
                    for idx, msg in enumerate(messages) for rec in receiver_ids
                ]
                best_role = max(role_rank) if role_rank else None
                role_key = (best_role[-2], best_role[-1]) if best_role and best_role[0] > 0 else None
                
                # 4. Full
                full_key = (0, receiver_ids[0]) if messages and receiver_ids else None
                
                # 5. Fixed-frequency
                fixed_freq = int(config["communication"].get("fixed_frequency", 3))
                fixed_key = role_key if int(context["step"]) % fixed_freq == 0 else None
                
                # 6. Never
                never_key = None
                
                # 7. Random
                rng = random.Random(int(context["seed"]) * 1009 + int(context["step"]))
                random_key = (rng.randrange(len(messages)), receiver_ids[0]) if messages and receiver_ids else None
                
                selected_keys = {
                    "aware": aware_key,
                    "agnostic": agnostic_key,
                    "role_only": role_key,
                    "fixed_frequency": fixed_key,
                    "full": full_key,
                    "never": never_key,
                    "random": random_key,
                }
                
                for m, key in selected_keys.items():
                    choice = (messages[key[0]], key[1]) if key is not None else None
                    ret = rollout_task_return(context, choice, config)
                    oracle_best = max([0.0, *truth.values()]) if truth else 0.0
                    actual_oracle = truth.get(key, 0.0) if key is not None else 0.0
                    regret = oracle_best - actual_oracle
                    
                    method_rows[m].append({
                        "seed": context["seed"],
                        "step": context["step"],
                        "messages": 1 if key is not None else 0,
                        "return": ret,
                        "regret": regret,
                    })
                    
            # Compute stats for this (gamma, lambda)
            method_stats = {}
            for m in methods:
                rets = [r["return"] for r in method_rows[m]]
                msgs = [r["messages"] for r in method_rows[m]]
                regs = [r["regret"] for r in method_rows[m]]
                
                mean_ret = mean(rets)
                mean_msg = mean(msgs)
                mean_reg = mean(regs)
                
                method_stats[m] = {
                    "mean_return": mean_ret,
                    "mean_messages": mean_msg,
                    "mean_regret": mean_reg,
                }
                
                pareto_points.append({
                    "gamma": gamma,
                    "lambda": lam,
                    "method": m,
                    "mean_return": mean_ret,
                    "mean_messages": mean_msg,
                })
                
                csv_rows.append({
                    "gamma": gamma,
                    "lambda": lam,
                    "method": m,
                    "mean_return": round(mean_ret, 4),
                    "mean_messages": round(mean_msg, 4),
                    "mean_regret": round(mean_reg, 4),
                })
                
            # Paired aware vs agnostic difference
            paired_by_seed = defaultdict(list)
            for aware_row, agnostic_row in zip(method_rows["aware"], method_rows["agnostic"]):
                paired_by_seed[aware_row["seed"]].append(aware_row["return"] - agnostic_row["return"])
            paired_diffs = [mean(differences) for differences in paired_by_seed.values()]
            paired_ci = bootstrap_mean_interval(paired_diffs, confidence, bootstrap_samples, seed=42)
            raw_rows.extend(
                {"gamma": gamma, "lambda": lam, "method": method, **row}
                for method, rows in method_rows.items() for row in rows
            )

            records_by_lambda[lam] = {
                "methods": method_stats,
                "aware_minus_agnostic": {
                    "mean_difference": mean(paired_diffs),
                    "confidence_interval": paired_ci,
                    "significant": paired_ci[0] > 0 or paired_ci[1] < 0,
                }
            }
            
        # Determine Pareto frontier for this gamma
        frontier = [
            pt for pt in pareto_points if not any(
                other["mean_messages"] <= pt["mean_messages"]
                and other["mean_return"] >= pt["mean_return"]
                and (other["mean_messages"] < pt["mean_messages"] or other["mean_return"] > pt["mean_return"])
                for other in pareto_points
            )
        ]
        
        results_by_gamma[gamma] = {
            "by_lambda": records_by_lambda,
            "pareto_frontier": sorted(frontier, key=lambda x: (x["mean_messages"], -x["mean_return"])),
            "runtime_seconds": time.perf_counter() - gamma_start,
        }
        print(f"Finished Gamma {gamma} in {time.perf_counter() - gamma_start:.1f}s.")
        
    stamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S_%f")
    out_dir = Path("results/sweep_gamma_lambda") / f"run_{stamp}"
    out_dir.mkdir(parents=True, exist_ok=False)
    (out_dir / "contexts.json").write_text(json.dumps(raw_rows, indent=2), encoding="utf-8")
    (out_dir / "config_used.json").write_text(json.dumps(base_config, indent=2), encoding="utf-8")
    
    summary_data = {
        "title": "VoC Cross-Parameter Evaluation: Discount Factors (gamma) and Cost Weights (lambda)",
        "gammas": gammas,
        "lambdas": lambdas,
        "methods": methods,
        "results_by_gamma": results_by_gamma,
        "total_runtime_seconds": time.perf_counter() - started_at,
    }
    
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)
        
    import csv
    with open(out_dir / "summary.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["gamma", "lambda", "method", "mean_return", "mean_messages", "mean_regret"])
        writer.writeheader()
        writer.writerows(csv_rows)
        
    print(f"Sweep completely written to {out_dir / 'summary.json'} and summary.csv.")
    return summary_data


if __name__ == "__main__":
    run_sweep()

