"""
run_ablation_attention_indonesia.py

Menjalankan studi ablasi lengkap untuk Attention Multi-Agent MO-CL-D3QN.

Varian:
1. full_attention_mo_cl_d3qn
2. attention_ablation_no_cl
3. attention_ablation_single_objective
4. ablation_no_attention

Contoh:
python run_ablation_attention_indonesia.py --episodes 120 --sumocfg city1_indonesia_2lane_leftfree.sumocfg

Uji cepat:
python run_ablation_attention_indonesia.py --episodes 5 --max-steps 500
"""

import argparse
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

from train_multiagent_mo_cl_d3qn_indonesia import Config
from train_multiagent_attention_mo_cl_d3qn_indonesia import train_attention


DEFAULT_VARIANTS = [
    "full_attention_mo_cl_d3qn",
    "attention_ablation_no_cl",
    "attention_ablation_single_objective",
    "ablation_no_attention",
]


def load_all(output_dir, variants):
    frames = []
    for v in variants:
        p = Path(output_dir) / v / "training_metrics.csv"
        if p.exists():
            frames.append(pd.read_csv(p))
    if not frames:
        raise FileNotFoundError("training_metrics.csv belum ditemukan.")
    return pd.concat(frames, ignore_index=True)


def summarize_and_plot(df, output_dir, last_n=20):
    output_dir = Path(output_dir)
    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    metrics = [
        ("avg_waiting_time", "Average Waiting Time (s)", True),
        ("avg_queue_length", "Average Queue Length", True),
        ("throughput", "Throughput", False),
        ("fuel_consumption_liter_est", "Fuel Consumption (L)", True),
        ("avg_speed", "Average Speed (m/s)", False),
        ("avg_travel_time", "Average Travel Time (s)", True),
        ("cumulative_reward", "Cumulative Reward", False),
        ("avg_loss", "Training Loss", True),
    ]

    last_df = df.sort_values("episode").groupby("variant").tail(last_n)
    summary = last_df.groupby("variant").agg(
        mean_reward=("cumulative_reward", "mean"),
        mean_awt_s=("avg_waiting_time", "mean"),
        mean_aql=("avg_queue_length", "mean"),
        mean_throughput=("throughput", "mean"),
        mean_fuel_L=("fuel_consumption_liter_est", "mean"),
        mean_speed=("avg_speed", "mean"),
        mean_att_s=("avg_travel_time", "mean"),
        mean_loss=("avg_loss", "mean"),
    ).round(4).reset_index()
    summary.to_csv(output_dir / f"attention_ablation_last{last_n}_summary.csv", index=False)

    for col, ylabel, _ in metrics:
        plt.figure(figsize=(11, 6))
        for variant, g in df.groupby("variant"):
            g = g.sort_values("episode")
            y = g[col].rolling(5, min_periods=1).mean()
            plt.plot(g["episode"], y, linewidth=2, label=variant)
        plt.axvline(40, linestyle="--", linewidth=1)
        plt.axvline(80, linestyle="--", linewidth=1)
        plt.xlabel("Episode")
        plt.ylabel(ylabel)
        plt.title(f"Attention Ablation - {ylabel}")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(plot_dir / f"{col}_curve.png", dpi=300, bbox_inches="tight")
        plt.close()

    # Bar chart last-N
    for col, ylabel, _ in metrics[:6]:
        map_col = {
            "avg_waiting_time": "mean_awt_s",
            "avg_queue_length": "mean_aql",
            "throughput": "mean_throughput",
            "fuel_consumption_liter_est": "mean_fuel_L",
            "avg_speed": "mean_speed",
            "avg_travel_time": "mean_att_s",
        }[col]
        plt.figure(figsize=(10, 5))
        plt.bar(summary["variant"], summary[map_col])
        plt.xlabel("Variant")
        plt.ylabel(ylabel)
        plt.title(f"Rata-rata {ylabel} pada {last_n} Episode Terakhir")
        plt.xticks(rotation=20, ha="right")
        plt.grid(True, axis="y", alpha=0.3)
        plt.tight_layout()
        plt.savefig(plot_dir / f"bar_last{last_n}_{col}.png", dpi=300, bbox_inches="tight")
        plt.close()

    return summary


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--sumocfg", default="city1_indonesia_2lane_leftfree.sumocfg")
    p.add_argument("--episodes", type=int, default=120)
    p.add_argument("--max-steps", type=int, default=3600)
    p.add_argument("--output-dir", default="outputs_attention_indonesia")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--stage1-end", type=int, default=40)
    p.add_argument("--stage2-end", type=int, default=80)
    p.add_argument("--decision-interval", type=int, default=10)
    p.add_argument("--yellow-duration", type=int, default=3)
    p.add_argument("--top-k-neighbors", type=int, default=4)
    p.add_argument("--attention-dim", type=int, default=128)
    p.add_argument("--last-n", type=int, default=20)
    p.add_argument("--gui", action="store_true")
    p.add_argument("--plot-only", action="store_true")
    p.add_argument("--variants", nargs="+", default=DEFAULT_VARIANTS)
    return p.parse_args()


def main():
    a = parse_args()

    if not a.plot_only:
        for i, variant in enumerate(a.variants):
            cfg = Config(
                sumocfg=a.sumocfg,
                output_dir=a.output_dir,
                episodes=a.episodes,
                max_steps=a.max_steps,
                seed=a.seed + i * 1000,
                stage1_end=a.stage1_end,
                stage2_end=a.stage2_end,
                decision_interval=a.decision_interval,
                yellow_duration=a.yellow_duration,
            )
            train_attention(
                cfg,
                variant,
                gui=a.gui,
                top_k_neighbors=a.top_k_neighbors,
                attention_dim=a.attention_dim,
                log_attention_every=20,
            )

    df = load_all(a.output_dir, a.variants)
    df.to_csv(Path(a.output_dir) / "attention_ablation_all_metrics.csv", index=False)
    summary = summarize_and_plot(df, a.output_dir, last_n=a.last_n)

    print("=" * 80)
    print("Studi ablasi Attention MO-CL-D3QN selesai.")
    print(f"Output: {a.output_dir}")
    print(summary.to_string(index=False))
    print("=" * 80)


if __name__ == "__main__":
    main()
