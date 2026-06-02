"""
Visualisasi hasil training dan ablation study multi-intersection.
"""

import argparse
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


def load_metrics(output_dir: Path):
    frames = []
    for p in output_dir.glob("*/training_metrics.csv"):
        frames.append(pd.read_csv(p))
    if not frames:
        raise FileNotFoundError("Tidak ada training_metrics.csv ditemukan.")
    return pd.concat(frames, ignore_index=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="outputs_multi_indonesia_ablation")
    parser.add_argument("--last-n", type=int, default=20)
    args = parser.parse_args()

    out = Path(args.output_dir)
    plots = out / "plots"
    plots.mkdir(parents=True, exist_ok=True)
    df = load_metrics(out)
    df.to_csv(out / "all_metrics.csv", index=False)

    metrics = [
        ("cumulative_reward", "Cumulative Reward", "01_reward.png"),
        ("avg_waiting_time", "Average Waiting Time (s)", "02_awt.png"),
        ("avg_queue_length", "Average Queue Length", "03_aql.png"),
        ("throughput", "Throughput", "04_throughput.png"),
        ("fuel_consumption_liter_est", "Fuel Consumption (L)", "05_fuel.png"),
        ("avg_speed", "Average Speed (m/s)", "06_speed.png"),
        ("avg_travel_time", "Average Travel Time (s)", "07_att.png"),
        ("avg_loss", "Training Loss", "08_loss.png"),
    ]
    for col, ylabel, fname in metrics:
        plt.figure(figsize=(11,6))
        for variant, g in df.groupby("variant"):
            g = g.sort_values("episode").copy()
            g[col+"_ma5"] = g[col].rolling(5, min_periods=1).mean()
            plt.plot(g["episode"], g[col+"_ma5"], linewidth=2, label=variant)
        plt.axvline(40, linestyle="--", linewidth=1)
        plt.axvline(80, linestyle="--", linewidth=1)
        plt.xlabel("Episode")
        plt.ylabel(ylabel)
        plt.title(f"Multi-Intersection Indonesia - {ylabel}")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(plots / fname, dpi=300, bbox_inches="tight")
        plt.close()

    last = df.sort_values("episode").groupby("variant").tail(args.last_n)
    summary = last.groupby("variant").agg(
        mean_reward=("cumulative_reward", "mean"),
        mean_awt_s=("avg_waiting_time", "mean"),
        mean_aql=("avg_queue_length", "mean"),
        mean_throughput=("throughput", "mean"),
        mean_fuel_L=("fuel_consumption_liter_est", "mean"),
        mean_speed=("avg_speed", "mean"),
        mean_att_s=("avg_travel_time", "mean"),
        mean_loss=("avg_loss", "mean"),
    ).round(4).reset_index()
    summary.to_csv(out / f"ablation_last{args.last_n}_simple_summary.csv", index=False)

    bar_metrics = [
        ("mean_awt_s", "Average Waiting Time (s)", "09_bar_awt.png"),
        ("mean_aql", "Average Queue Length", "10_bar_aql.png"),
        ("mean_throughput", "Throughput", "11_bar_throughput.png"),
        ("mean_fuel_L", "Fuel Consumption (L)", "12_bar_fuel.png"),
        ("mean_speed", "Average Speed (m/s)", "13_bar_speed.png"),
        ("mean_att_s", "Average Travel Time (s)", "14_bar_att.png"),
    ]
    for col, ylabel, fname in bar_metrics:
        plt.figure(figsize=(9,5))
        plt.bar(summary["variant"], summary[col])
        plt.xticks(rotation=20, ha="right")
        plt.xlabel("Variant")
        plt.ylabel(ylabel)
        plt.title(f"Rata-rata {ylabel} pada {args.last_n} Episode Terakhir")
        plt.grid(True, axis="y", alpha=0.3)
        plt.tight_layout()
        plt.savefig(plots / fname, dpi=300, bbox_inches="tight")
        plt.close()

    print("Visualisasi selesai.")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
