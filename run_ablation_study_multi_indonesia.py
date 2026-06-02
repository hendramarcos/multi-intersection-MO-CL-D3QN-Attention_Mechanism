"""
Menjalankan studi ablasi lengkap untuk multi-intersection MO-CL-D3QN Indonesia.

Varian:
- full_mo_cl_d3qn
- ablation_no_cl
- ablation_single_objective

Contoh:
python run_ablation_study_multi_indonesia.py --episodes 120
python run_ablation_study_multi_indonesia.py --episodes 5 --max-steps 500
"""

import argparse
import subprocess
import sys
from pathlib import Path


def run(cmd):
    print("\n" + "="*80)
    print("Menjalankan:", " ".join(map(str, cmd)))
    print("="*80)
    proc = subprocess.run(cmd)
    if proc.returncode != 0:
        raise RuntimeError("Perintah gagal")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=int, default=120)
    p.add_argument("--max-steps", type=int, default=3600)
    p.add_argument("--sumocfg", default="city1_indonesia_2lane_leftfree.sumocfg")
    p.add_argument("--output-dir", default="outputs_multi_indonesia_ablation")
    p.add_argument("--gui", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--variants", nargs="+", default=["full_mo_cl_d3qn", "ablation_no_cl", "ablation_single_objective"])
    args = p.parse_args()

    for i, variant in enumerate(args.variants):
        cmd = [sys.executable, "train_multiagent_mo_cl_d3qn_indonesia.py",
               "--sumocfg", args.sumocfg,
               "--variant", variant,
               "--episodes", str(args.episodes),
               "--max-steps", str(args.max_steps),
               "--output-dir", args.output_dir,
               "--seed", str(args.seed + i*1000)]
        if args.gui:
            cmd.append("--gui")
        run(cmd)

    run([sys.executable, "plot_results_multi_indonesia.py", "--output-dir", args.output_dir])
    print("\nStudi ablasi selesai.")
    print(f"Output: {Path(args.output_dir).resolve()}")


if __name__ == "__main__":
    main()
