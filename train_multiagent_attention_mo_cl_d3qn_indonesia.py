"""
train_multiagent_attention_mo_cl_d3qn_indonesia.py

Multi-Agent Attention MO-CL-D3QN untuk jaringan multi-simpang SUMO konteks Indonesia:
- kendaraan berjalan di lajur kiri / left-hand traffic
- setir kanan / right-hand steering context
- koordinasi antar-persimpangan menggunakan Attention Mechanism
- setiap traffic light agent menerima local state + state tetangga
- aksi agen memilih fase hijau; jika fase yang sama dipilih berulang, durasi hijau otomatis diperpanjang
- objective: memaksimalkan throughput serta meminimalkan waiting time, queue length, travel time/delay proxy, dan fuel consumption
- curriculum learning bertahap
- mendukung studi ablasi:
  1. full_attention_mo_cl_d3qn
  2. attention_ablation_no_cl
  3. attention_ablation_single_objective
  4. ablation_no_attention

Contoh training:
python train_multiagent_attention_mo_cl_d3qn_indonesia.py --episodes 120 --variant full_attention_mo_cl_d3qn --sumocfg city1_indonesia_2lane_leftfree.sumocfg

Uji cepat:
python train_multiagent_attention_mo_cl_d3qn_indonesia.py --episodes 5 --max-steps 500 --variant full_attention_mo_cl_d3qn
"""

import argparse
import csv
import math
import os
import random
import sys
import xml.etree.ElementTree as ET
from collections import deque, namedtuple
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

try:
    if "SUMO_HOME" in os.environ:
        tools = os.path.join(os.environ["SUMO_HOME"], "tools")
        if tools not in sys.path:
            sys.path.append(tools)
    import traci
    import sumolib
except Exception as exc:
    raise RuntimeError("SUMO/TraCI/sumolib tidak ditemukan. Pastikan SUMO_HOME sudah diset.") from exc

from train_multiagent_mo_cl_d3qn_indonesia import (
    Config,
    MultiIntersectionEnv,
    set_seed,
    append_csv,
)


AttentionTransition = namedtuple(
    "AttentionTransition",
    ["local", "neighbors", "mask", "action", "reward", "next_local", "next_neighbors", "next_mask", "done"]
)


def stage_of_episode_attention(ep: int, cfg: Config, variant: str) -> int:
    if variant in ["attention_ablation_no_cl"]:
        return 3
    if ep <= cfg.stage1_end:
        return 1
    if ep <= cfg.stage2_end:
        return 2
    return 3


def env_variant_from_attention_variant(variant: str) -> str:
    if variant == "attention_ablation_no_cl":
        return "ablation_no_cl"
    if variant == "attention_ablation_single_objective":
        return "ablation_single_objective"
    # ablation_no_attention tetap memakai reward full, tetapi neighbor context dikosongkan
    return "full_mo_cl_d3qn"


class AttentionReplayBuffer:
    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)

    def push(self, *args):
        self.buffer.append(AttentionTransition(*args))

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        return AttentionTransition(*zip(*batch))

    def __len__(self):
        return len(self.buffer)


class AttentionDuelingDQN(nn.Module):
    """
    Dueling DQN dengan attention antar-agent.
    Input:
      local_state   : [B, state_dim]
      neighbor_state: [B, K, state_dim]
      mask          : [B, K], 1 untuk neighbor valid, 0 untuk padding

    Mekanisme:
      query berasal dari local state.
      key dan value berasal dari state tetangga.
      context vector = weighted sum value tetangga.
      Q-value dihasilkan dari local embedding + attention context.
    """

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256, attention_dim: int = 128):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.attention_dim = attention_dim

        self.local_encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, attention_dim),
            nn.ReLU(),
        )
        self.neighbor_encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, attention_dim),
            nn.ReLU(),
        )

        self.query = nn.Linear(attention_dim, attention_dim)
        self.key = nn.Linear(attention_dim, attention_dim)
        self.value = nn.Linear(attention_dim, attention_dim)

        fused_dim = attention_dim * 2
        self.fusion = nn.Sequential(
            nn.Linear(fused_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.value_stream = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )
        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, action_dim),
        )

    def forward(self, local_state, neighbor_states, mask=None, return_attention=False):
        local_emb = self.local_encoder(local_state)        # [B, D]
        neigh_emb = self.neighbor_encoder(neighbor_states) # [B, K, D]

        q = self.query(local_emb).unsqueeze(1)             # [B, 1, D]
        k = self.key(neigh_emb)                            # [B, K, D]
        v = self.value(neigh_emb)                          # [B, K, D]

        scores = torch.sum(q * k, dim=-1) / math.sqrt(self.attention_dim)  # [B, K]

        if mask is not None:
            # padding neighbor diberi nilai sangat kecil agar tidak dipilih
            scores = scores.masked_fill(mask <= 0, -1e9)

            # bila semua mask 0, softmax bisa menjadi NaN. Tangani dengan konteks nol.
            valid_count = mask.sum(dim=1, keepdim=True)
            all_invalid = valid_count <= 0
        else:
            all_invalid = None

        attn = torch.softmax(scores, dim=-1)  # [B, K]

        if mask is not None:
            attn = attn * mask
            denom = attn.sum(dim=1, keepdim=True).clamp_min(1e-8)
            attn = attn / denom
            if all_invalid is not None and all_invalid.any():
                attn = torch.where(all_invalid, torch.zeros_like(attn), attn)

        context = torch.bmm(attn.unsqueeze(1), v).squeeze(1)  # [B, D]
        fused = torch.cat([local_emb, context], dim=-1)
        features = self.fusion(fused)

        value = self.value_stream(features)
        advantage = self.advantage_stream(features)
        q_values = value + advantage - advantage.mean(dim=1, keepdim=True)

        if return_attention:
            return q_values, attn
        return q_values


class AttentionD3QNAgent:
    def __init__(self, state_dim: int, action_dim: int, cfg: Config, device, attention_dim: int = 128):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.cfg = cfg
        self.device = device
        self.policy_net = AttentionDuelingDQN(state_dim, action_dim, cfg.hidden_dim, attention_dim).to(device)
        self.target_net = AttentionDuelingDQN(state_dim, action_dim, cfg.hidden_dim, attention_dim).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=cfg.learning_rate)
        self.replay = AttentionReplayBuffer(cfg.replay_size)
        self.train_steps = 0

    def epsilon(self):
        frac = min(1.0, self.train_steps / self.cfg.epsilon_decay_steps)
        return self.cfg.epsilon_start + frac * (self.cfg.epsilon_end - self.cfg.epsilon_start)

    def select_action(self, local_state, neighbor_states, mask, eval_mode=False):
        eps = 0.0 if eval_mode else self.epsilon()
        if random.random() < eps:
            return random.randrange(self.action_dim)
        with torch.no_grad():
            local_t = torch.as_tensor(local_state, dtype=torch.float32, device=self.device).unsqueeze(0)
            neigh_t = torch.as_tensor(neighbor_states, dtype=torch.float32, device=self.device).unsqueeze(0)
            mask_t = torch.as_tensor(mask, dtype=torch.float32, device=self.device).unsqueeze(0)
            q = self.policy_net(local_t, neigh_t, mask_t)
            return int(torch.argmax(q, dim=1).item())

    def get_attention_weights(self, local_state, neighbor_states, mask):
        with torch.no_grad():
            local_t = torch.as_tensor(local_state, dtype=torch.float32, device=self.device).unsqueeze(0)
            neigh_t = torch.as_tensor(neighbor_states, dtype=torch.float32, device=self.device).unsqueeze(0)
            mask_t = torch.as_tensor(mask, dtype=torch.float32, device=self.device).unsqueeze(0)
            _, attn = self.policy_net(local_t, neigh_t, mask_t, return_attention=True)
            return attn.squeeze(0).detach().cpu().numpy()

    def optimize(self):
        if len(self.replay) < max(self.cfg.min_replay_size, self.cfg.batch_size):
            return None

        batch = self.replay.sample(self.cfg.batch_size)

        local = torch.as_tensor(np.array(batch.local), dtype=torch.float32, device=self.device)
        neighbors = torch.as_tensor(np.array(batch.neighbors), dtype=torch.float32, device=self.device)
        mask = torch.as_tensor(np.array(batch.mask), dtype=torch.float32, device=self.device)
        actions = torch.as_tensor(batch.action, dtype=torch.long, device=self.device).unsqueeze(1)
        rewards = torch.as_tensor(batch.reward, dtype=torch.float32, device=self.device).unsqueeze(1)
        next_local = torch.as_tensor(np.array(batch.next_local), dtype=torch.float32, device=self.device)
        next_neighbors = torch.as_tensor(np.array(batch.next_neighbors), dtype=torch.float32, device=self.device)
        next_mask = torch.as_tensor(np.array(batch.next_mask), dtype=torch.float32, device=self.device)
        dones = torch.as_tensor(batch.done, dtype=torch.float32, device=self.device).unsqueeze(1)

        q_pred = self.policy_net(local, neighbors, mask).gather(1, actions)

        with torch.no_grad():
            next_actions = torch.argmax(self.policy_net(next_local, next_neighbors, next_mask), dim=1, keepdim=True)
            next_q = self.target_net(next_local, next_neighbors, next_mask).gather(1, next_actions)
            q_target = rewards + self.cfg.gamma * (1.0 - dones) * next_q

        loss = nn.SmoothL1Loss()(q_pred, q_target)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy_net.parameters(), 10.0)
        self.optimizer.step()

        self.train_steps += 1
        if self.train_steps % self.cfg.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        return float(loss.item())

    def save(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "policy_state_dict": self.policy_net.state_dict(),
            "target_state_dict": self.target_net.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "train_steps": self.train_steps,
            "state_dim": self.state_dim,
            "action_dim": self.action_dim,
        }, path)

    def load(self, path):
        ckpt = torch.load(path, map_location=self.device)
        self.policy_net.load_state_dict(ckpt["policy_state_dict"])
        self.target_net.load_state_dict(ckpt.get("target_state_dict", ckpt["policy_state_dict"]))
        if "optimizer_state_dict" in ckpt:
            self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        self.train_steps = ckpt.get("train_steps", 0)


def get_net_file_from_sumocfg(sumocfg: str) -> Path:
    cfg_path = Path(sumocfg)
    tree = ET.parse(cfg_path)
    root = tree.getroot()
    net_elem = root.find(".//net-file")
    if net_elem is None:
        raise RuntimeError("Tidak menemukan <net-file> pada sumocfg.")
    net_value = net_elem.attrib.get("value")
    return (cfg_path.parent / net_value).resolve()


def build_neighbor_map_from_network(sumocfg: str, tls_ids: List[str], top_k: int = 4) -> Dict[str, List[str]]:
    """
    Membangun tetangga berdasarkan jarak geometris antar node traffic light.
    Cocok untuk city network multi-simpang dan tidak bergantung pada nama edge.
    """
    try:
        net_path = get_net_file_from_sumocfg(sumocfg)
        net = sumolib.net.readNet(str(net_path))
        coords = {}
        for tls in tls_ids:
            try:
                node = net.getNode(tls)
                coords[tls] = node.getCoord()
            except Exception:
                pass

        if len(coords) < 2:
            raise RuntimeError("Koordinat TLS kurang.")

        neighbor_map = {}
        for tls in tls_ids:
            if tls not in coords:
                neighbor_map[tls] = [x for x in tls_ids if x != tls][:top_k]
                continue
            x1, y1 = coords[tls]
            dist = []
            for other in tls_ids:
                if other == tls or other not in coords:
                    continue
                x2, y2 = coords[other]
                d = math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
                dist.append((d, other))
            dist.sort()
            neighbor_map[tls] = [o for _, o in dist[:top_k]]
        return neighbor_map
    except Exception:
        # fallback: semua TLS selain dirinya, dibatasi top_k
        return {tls: [x for x in tls_ids if x != tls][:top_k] for tls in tls_ids}


def build_attention_bundle(
    tls: str,
    state_dict: Dict[str, np.ndarray],
    neighbor_map: Dict[str, List[str]],
    top_k: int,
    state_dim: int,
    no_attention: bool = False,
):
    local_state = state_dict[tls]
    neighbor_states = np.zeros((top_k, state_dim), dtype=np.float32)
    mask = np.zeros(top_k, dtype=np.float32)

    if no_attention:
        return local_state, neighbor_states, mask

    neighbors = neighbor_map.get(tls, [])[:top_k]
    for i, nb in enumerate(neighbors):
        if nb in state_dict:
            neighbor_states[i] = state_dict[nb]
            mask[i] = 1.0
    return local_state, neighbor_states, mask


def log_attention_weights(path, episode, decision, tls, neighbors, weights):
    row = {"episode": episode, "decision": decision, "tls": tls}
    for i, nb in enumerate(neighbors):
        row[f"neighbor_{i+1}"] = nb
        row[f"weight_{i+1}"] = float(weights[i]) if i < len(weights) else 0.0
    append_csv(path, row)


def train_attention(cfg: Config, variant: str, gui=False, top_k_neighbors=4, attention_dim=128, log_attention_every=20):
    set_seed(cfg.seed)

    env_variant = env_variant_from_attention_variant(variant)
    no_attention = variant == "ablation_no_attention"

    env = MultiIntersectionEnv(cfg, env_variant, gui)
    initial_states = env.reset(seed=cfg.seed)
    neighbor_map = build_neighbor_map_from_network(cfg.sumocfg, env.tls_ids, top_k=top_k_neighbors)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agent = AttentionD3QNAgent(env.state_dim, env.action_dim, cfg, device, attention_dim=attention_dim)
    env.close()

    out = Path(cfg.output_dir) / variant
    csv_path = out / "training_metrics.csv"
    attention_path = out / "attention_weights.csv"
    neighbor_path = out / "neighbor_map.csv"
    best_score = -float("inf")

    for tls, nbs in neighbor_map.items():
        append_csv(neighbor_path, {"tls": tls, "neighbors": " ".join(nbs)})

    print("=" * 80)
    print("Attention Multi-Agent MO-CL-D3QN")
    print("Konteks: Indonesia | lajur kiri | setir kanan | koordinasi antar-simpang")
    print(f"Variant: {variant} | Env reward variant: {env_variant} | No attention: {no_attention}")
    print(f"TLS agents: {len(env.tls_ids)} | Action dim: {env.action_dim} | Local state dim: {env.state_dim}")
    print(f"Top-K neighbors: {top_k_neighbors} | Attention dim: {attention_dim} | Device: {device}")
    print(f"Output: {out}")
    print("=" * 80)

    try:
        for ep in range(1, cfg.episodes + 1):
            stage = stage_of_episode_attention(ep, cfg, variant)
            states = env.reset(seed=cfg.seed + ep)
            done = False
            decision = 0

            while not done:
                bundles = {
                    tls: build_attention_bundle(
                        tls, states, neighbor_map, top_k_neighbors, env.state_dim, no_attention=no_attention
                    )
                    for tls in env.tls_ids
                }

                actions = {
                    tls: agent.select_action(local, neigh, mask, eval_mode=False)
                    for tls, (local, neigh, mask) in bundles.items()
                }

                next_states, rewards, done, _ = env.step(actions, stage)

                next_bundles = {
                    tls: build_attention_bundle(
                        tls, next_states, neighbor_map, top_k_neighbors, env.state_dim, no_attention=no_attention
                    )
                    for tls in env.tls_ids
                }

                for tls in env.tls_ids:
                    local, neigh, mask = bundles[tls]
                    next_local, next_neigh, next_mask = next_bundles[tls]
                    agent.replay.push(local, neigh, mask, actions[tls], rewards[tls], next_local, next_neigh, next_mask, done)
                    loss = agent.optimize()
                    if loss is not None:
                        env.metrics["loss_sum"] += loss
                        env.metrics["loss_count"] += 1

                # Log attention weight secara periodik untuk interpretabilitas koordinasi antar-simpang.
                if (not no_attention) and log_attention_every > 0 and decision % log_attention_every == 0:
                    for tls in env.tls_ids:
                        local, neigh, mask = bundles[tls]
                        weights = agent.get_attention_weights(local, neigh, mask)
                        nbs = neighbor_map.get(tls, [])[:top_k_neighbors]
                        log_attention_weights(attention_path, ep, decision, tls, nbs, weights)

                states = next_states
                decision += 1

            summ = env.summary()
            row = {"variant": variant, "episode": ep, "stage": stage, "epsilon": agent.epsilon(), **summ}
            append_csv(csv_path, row)

            # Skor model tetap menggunakan reward kumulatif varian tersebut.
            # Untuk laporan ilmiah, metrik lalu lintas adalah pembanding utama.
            if summ["cumulative_reward"] > best_score:
                best_score = summ["cumulative_reward"]
                agent.save(out / "best_model.pt")
            agent.save(out / "last_model.pt")

            print(
                f"{variant} | EP {ep:03d} | Stage {stage} | R {summ['cumulative_reward']:.3f} | "
                f"AWT {summ['avg_waiting_time']:.2f}s | AQL {summ['avg_queue_length']:.2f} | "
                f"TP {summ['throughput']} | Fuel {summ['fuel_consumption_liter_est']:.3f}L | "
                f"Speed {summ['avg_speed']:.2f} | ATT {summ['avg_travel_time']:.2f}s | "
                f"Loss {summ['avg_loss']:.5f} | eps {agent.epsilon():.3f}"
            )
    finally:
        env.close()

    print(f"Selesai. Log: {csv_path}")
    print(f"Model terbaik: {out / 'best_model.pt'}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--sumocfg", default="city1_indonesia_2lane_leftfree.sumocfg")
    p.add_argument("--variant", default="full_attention_mo_cl_d3qn",
                   choices=[
                       "full_attention_mo_cl_d3qn",
                       "attention_ablation_no_cl",
                       "attention_ablation_single_objective",
                       "ablation_no_attention",
                   ])
    p.add_argument("--episodes", type=int, default=120)
    p.add_argument("--max-steps", type=int, default=3600)
    p.add_argument("--gui", action="store_true")
    p.add_argument("--output-dir", default="outputs_attention_indonesia")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--stage1-end", type=int, default=40)
    p.add_argument("--stage2-end", type=int, default=80)
    p.add_argument("--decision-interval", type=int, default=10)
    p.add_argument("--yellow-duration", type=int, default=3)
    p.add_argument("--top-k-neighbors", type=int, default=4)
    p.add_argument("--attention-dim", type=int, default=128)
    p.add_argument("--log-attention-every", type=int, default=20)
    return p.parse_args()


def main():
    a = parse_args()
    cfg = Config(
        sumocfg=a.sumocfg,
        output_dir=a.output_dir,
        episodes=a.episodes,
        max_steps=a.max_steps,
        seed=a.seed,
        stage1_end=a.stage1_end,
        stage2_end=a.stage2_end,
        decision_interval=a.decision_interval,
        yellow_duration=a.yellow_duration,
    )
    train_attention(
        cfg,
        a.variant,
        gui=a.gui,
        top_k_neighbors=a.top_k_neighbors,
        attention_dim=a.attention_dim,
        log_attention_every=a.log_attention_every,
    )


if __name__ == "__main__":
    main()
