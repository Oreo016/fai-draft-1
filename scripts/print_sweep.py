import argparse
import json
from pathlib import Path

parser = argparse.ArgumentParser(description="Print a saved gamma/lambda sweep.")
parser.add_argument("summary", nargs="?", type=Path)
arguments = parser.parse_args()
sweep_root = Path("results/sweep_gamma_lambda")
summary_path = arguments.summary or next(
    iter(sorted(sweep_root.glob("run_*/summary.json"), reverse=True)),
    sweep_root / "summary.json",
)
with summary_path.open(encoding="utf-8") as f:
    data = json.load(f)

print("=== SWEEP SUMMARY ===")
print("Total runtime:", round(data["total_runtime_seconds"], 2), "s")

for gamma, g_res in data["results_by_gamma"].items():
    print(f"\n================ GAMMA = {gamma} ================")
    print("Lambda | Aware Ret | Agnostic Ret | Diff (Aware - Agnostic) | 95% CI          | Sig?  | Aware Msgs | Full Ret | Full Msgs")
    print("-" * 115)
    for lam, l_res in g_res["by_lambda"].items():
        m = l_res["methods"]
        diff = l_res["aware_minus_agnostic"]
        mean_diff = round(diff["mean_difference"], 3)
        ci_str = f"[{diff['confidence_interval'][0]:.3f}, {diff['confidence_interval'][1]:.3f}]"
        sig_str = str(diff["significant"])
        aw_ret = m["aware"]["mean_return"]
        ag_ret = m["agnostic"]["mean_return"]
        aw_msg = m["aware"]["mean_messages"]
        fl_ret = m["full"]["mean_return"]
        fl_msg = m["full"]["mean_messages"]
        print(f"{float(lam):6.2f} | {aw_ret:9.3f} | {ag_ret:12.3f} | {mean_diff:23.3f} | {ci_str:15s} | {sig_str:5s} | {aw_msg:10.2f} | {fl_ret:8.3f} | {fl_msg:9.2f}")
    
    print(f"\n  Pareto Frontier for gamma = {gamma}:")
    for pt in g_res["pareto_frontier"]:
        print(f"    method={pt['method']:15s} lam={pt['lambda']:4.2f} msgs={pt['mean_messages']:4.2f} ret={pt['mean_return']:6.3f}")

