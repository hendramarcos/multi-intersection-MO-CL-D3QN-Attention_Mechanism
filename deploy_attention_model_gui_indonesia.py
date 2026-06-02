"""
deploy_attention_model_gui_indonesia.py

Implementasi model Attention MO-CL-D3QN ke SUMO GUI.

Contoh:
python deploy_attention_model_gui_indonesia.py --model outputs_attention_indonesia/full_attention_mo_cl_d3qn/best_model.pt --gui --wait-enter --keep-open
"""

import argparse
from pathlib import Path
import csv
import torch
import numpy as np

from train_multiagent_mo_cl_d3qn_indonesia import Config, MultiIntersectionEnv, set_seed
from train_multiagent_attention_mo_cl_d3qn_indonesia import (
    AttentionD3QNAgent,
    build_neighbor_map_from_network,
    build_attention_bundle,
    env_variant_from_attention_variant,
)


def append_csv(path, row):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--sumocfg", default="city1_indonesia_2lane_leftfree.sumocfg")
    p.add_argument("--variant", default="full_attention_mo_cl_d3qn",
                   choices=["full_attention_mo_cl_d3qn", "attention_ablation_no_cl",
                            "attention_ablation_single_objective", "ablation_no_attention"])
    p.add_argument("--gui", action="store_true")
    p.add_argument("--max-steps", type=int, default=3600)
    p.add_argument("--decision-interval", type=int, default=10)
    p.add_argument("--yellow-duration", type=int, default=3)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--top-k-neighbors", type=int, default=4)
    p.add_argument("--attention-dim", type=int, default=128)
    p.add_argument("--output", default="outputs_attention_deployment/deployment_metrics.csv")
    p.add_argument("--attention-output", default="outputs_attention_deployment/deployment_attention_weights.csv")
    p.add_argument("--print-every", type=int, default=5)
    p.add_argument("--wait-enter", action="store_true")
    p.add_argument("--keep-open", action="store_true")
    return p.parse_args()


def main():
    a = parse_args()
    set_seed(a.seed)

    cfg = Config(
        sumocfg=a.sumocfg,
        output_dir="outputs_attention_deployment",
        episodes=1,
        max_steps=a.max_steps,
        seed=a.seed,
        decision_interval=a.decision_interval,
        yellow_duration=a.yellow_duration,
    )

    env_variant = env_variant_from_attention_variant(a.variant)
    no_attention = a.variant == "ablation_no_attention"

    env = MultiIntersectionEnv(cfg, env_variant, gui=a.gui)
    states = env.reset(seed=a.seed)
    neighbor_map = build_neighbor_map_from_network(cfg.sumocfg, env.tls_ids, top_k=a.top_k_neighbors)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agent = AttentionD3QNAgent(env.state_dim, env.action_dim, cfg, device, attention_dim=a.attention_dim)
    agent.load(a.model)

    print("=" * 80)
    print("DEPLOYMENT ATTENTION MO-CL-D3QN KE SUMO GUI")
    print("=" * 80)
    print(f"Model: {a.model}")
    print(f"Variant: {a.variant}")
    print(f"TLS agents: {len(env.tls_ids)}")
    print(f"Top-K neighbors: {a.top_k_neighbors}")
    print(f"Device: {device}")
    print("=" * 80)

    if a.wait_enter:
        input("SUMO GUI sudah terbuka. Tekan ENTER untuk mulai menjalankan model attention...")

    done = False
    decision = 0

    try:
        while not done:
            bundles = {
                tls: build_attention_bundle(
                    tls, states, neighbor_map, a.top_k_neighbors, env.state_dim, no_attention=no_attention
                )
                for tls in env.tls_ids
            }
            actions = {
                tls: agent.select_action(local, neigh, mask, eval_mode=True)
                for tls, (local, neigh, mask) in bundles.items()
            }

            # log attention weights untuk analisis koordinasi
            if not no_attention:
                for tls in env.tls_ids:
                    local, neigh, mask = bundles[tls]
                    weights = agent.get_attention_weights(local, neigh, mask)
                    nbs = neighbor_map.get(tls, [])[:a.top_k_neighbors]
                    row = {"decision": decision, "tls": tls}
                    for i, nb in enumerate(nbs):
                        row[f"neighbor_{i+1}"] = nb
                        row[f"weight_{i+1}"] = float(weights[i]) if i < len(weights) else 0.0
                    append_csv(a.attention_output, row)

            next_states, rewards, done, _ = env.step(actions, stage=3)
            states = next_states
            decision += 1

            if decision % a.print_every == 0:
                s = env.summary()
                print(
                    f"Decision {decision:04d} | AWT {s['avg_waiting_time']:.2f}s | "
                    f"AQL {s['avg_queue_length']:.2f} | TP {s['throughput']} | "
                    f"Fuel {s['fuel_consumption_liter_est']:.3f}L | "
                    f"Speed {s['avg_speed']:.2f} | ATT {s['avg_travel_time']:.2f}s"
                )

        s = env.summary()
        row = {"model": a.model, "variant": a.variant, "decision_steps": decision, **s}
        append_csv(a.output, row)

        print("=" * 80)
        print("Deployment selesai.")
        for k, v in s.items():
            print(f"{k}: {v}")
        print(f"Log metrik: {a.output}")
        print(f"Log attention weights: {a.attention_output}")

    finally:
        if a.keep_open:
            input("Tekan ENTER untuk menutup SUMO/TraCI...")
        env.close()


if __name__ == "__main__":
    main()
