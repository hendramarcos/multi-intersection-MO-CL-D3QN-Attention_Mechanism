"""
train_multiagent_mo_cl_d3qn_indonesia.py

Multi-Agent MO-CL-D3QN untuk jaringan multi-simpang SUMO dengan konteks Indonesia:
- left-hand traffic: kendaraan berjalan di sisi kiri jalan
- 2 lajur per arah
- belok kiri langsung/permissive di setiap traffic light
- right-hand steering: stir/pengemudi di kanan
- route dinamis: mobil dan motor dari random trips

Model:
- Parameter-sharing Dueling Double DQN untuk semua traffic light agents
- Multi-objective reward: waiting time, queue length, throughput, fuel consumption
- Curriculum Learning bertahap
- Studi ablasi melalui argumen --variant

Varian:
1. full_mo_cl_d3qn
2. ablation_no_cl
3. ablation_single_objective

Contoh:
python train_multiagent_mo_cl_d3qn_indonesia.py --episodes 120 --variant full_mo_cl_d3qn
python train_multiagent_mo_cl_d3qn_indonesia.py --episodes 5 --max-steps 500 --gui
"""

import argparse
import csv
import os
import random
import sys
from collections import deque, namedtuple
from dataclasses import dataclass
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
    from sumolib import checkBinary
except Exception as exc:
    raise RuntimeError("SUMO/TraCI tidak ditemukan. Pastikan SUMO_HOME sudah diset.") from exc


@dataclass
class Config:
    sumocfg: str = "city1_indonesia_2lane_leftfree.sumocfg"
    output_dir: str = "outputs_multi_indonesia"
    episodes: int = 120
    max_steps: int = 3600
    decision_interval: int = 10
    yellow_duration: int = 3
    seed: int = 42

    # Curriculum stage
    stage1_end: int = 40
    stage2_end: int = 80

    # State normalization
    max_lane_vehicles: float = 25.0
    max_lane_waiting: float = 300.0
    max_speed: float = 13.89
    max_controlled_lanes: int = 12

    # Reward normalization
    wait_norm: float = 120.0
    queue_norm: float = 40.0
    throughput_norm: float = 25.0
    fuel_norm: float = 70000.0

    # D3QN
    gamma: float = 0.99
    learning_rate: float = 1e-4
    batch_size: int = 64
    replay_size: int = 80000
    min_replay_size: int = 1000
    target_update_freq: int = 500
    hidden_dim: int = 256

    # Exploration
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_steps: int = 35000


Transition = namedtuple("Transition", ["state", "action", "reward", "next_state", "done"])


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class ReplayBuffer:
    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)
    def push(self, *args):
        self.buffer.append(Transition(*args))
    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        return Transition(*zip(*batch))
    def __len__(self):
        return len(self.buffer)


class DuelingDQN(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int):
        super().__init__()
        self.feature = nn.Sequential(
            nn.Linear(state_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
        )
        self.value = nn.Sequential(nn.Linear(hidden_dim, hidden_dim // 2), nn.ReLU(), nn.Linear(hidden_dim // 2, 1))
        self.adv = nn.Sequential(nn.Linear(hidden_dim, hidden_dim // 2), nn.ReLU(), nn.Linear(hidden_dim // 2, action_dim))
    def forward(self, x):
        f = self.feature(x)
        v = self.value(f)
        a = self.adv(f)
        return v + a - a.mean(dim=1, keepdim=True)


class D3QNAgent:
    def __init__(self, state_dim: int, action_dim: int, cfg: Config, device):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.cfg = cfg
        self.device = device
        self.policy_net = DuelingDQN(state_dim, action_dim, cfg.hidden_dim).to(device)
        self.target_net = DuelingDQN(state_dim, action_dim, cfg.hidden_dim).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=cfg.learning_rate)
        self.replay = ReplayBuffer(cfg.replay_size)
        self.train_steps = 0
    def epsilon(self):
        frac = min(1.0, self.train_steps / self.cfg.epsilon_decay_steps)
        return self.cfg.epsilon_start + frac * (self.cfg.epsilon_end - self.cfg.epsilon_start)
    def select_action(self, state: np.ndarray, eval_mode=False):
        eps = 0.0 if eval_mode else self.epsilon()
        if random.random() < eps:
            return random.randrange(self.action_dim)
        with torch.no_grad():
            s = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            return int(torch.argmax(self.policy_net(s), dim=1).item())
    def optimize(self):
        if len(self.replay) < max(self.cfg.min_replay_size, self.cfg.batch_size):
            return None
        trans = self.replay.sample(self.cfg.batch_size)
        states = torch.as_tensor(np.array(trans.state), dtype=torch.float32, device=self.device)
        actions = torch.as_tensor(trans.action, dtype=torch.long, device=self.device).unsqueeze(1)
        rewards = torch.as_tensor(trans.reward, dtype=torch.float32, device=self.device).unsqueeze(1)
        next_states = torch.as_tensor(np.array(trans.next_state), dtype=torch.float32, device=self.device)
        dones = torch.as_tensor(trans.done, dtype=torch.float32, device=self.device).unsqueeze(1)
        q = self.policy_net(states).gather(1, actions)
        with torch.no_grad():
            next_actions = torch.argmax(self.policy_net(next_states), dim=1, keepdim=True)
            next_q = self.target_net(next_states).gather(1, next_actions)
            target = rewards + self.cfg.gamma * (1.0 - dones) * next_q
        loss = nn.SmoothL1Loss()(q, target)
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


class MultiIntersectionEnv:
    def __init__(self, cfg: Config, variant: str, gui=False):
        self.cfg = cfg
        self.variant = variant
        self.gui = gui
        self.sumo_binary = checkBinary("sumo-gui" if gui else "sumo")
        self.tls_ids: List[str] = []
        self.green_phases: Dict[str, List[int]] = {}
        self.yellow_after_green: Dict[str, Dict[int, int]] = {}
        self.controlled_lanes: Dict[str, List[str]] = {}
        self.current_green: Dict[str, int] = {}
        self.action_dim = 2
        self.state_dim = cfg.max_controlled_lanes * 4 + self.action_dim
        self.metrics = None
        self.vehicle_depart = {}
        self.travel_times = []
    def _cmd(self, seed):
        return [self.sumo_binary, "-c", self.cfg.sumocfg, "--start", "--quit-on-end", "--no-step-log", "true", "--seed", str(seed)]
    def reset(self, seed=42):
        if traci.isLoaded():
            traci.close(False)
        traci.start(self._cmd(seed))
        self._init_tls()
        self.vehicle_depart = {}
        self.travel_times = []
        self.metrics = {
            "sum_avg_waiting_time": 0.0,
            "sum_queue_length": 0.0,
            "sum_avg_speed": 0.0,
            "sum_fuel_mg": 0.0,
            "throughput": 0,
            "steps": 0,
            "cumulative_reward": 0.0,
            "loss_sum": 0.0,
            "loss_count": 0,
        }
        return self.get_all_states()
    def close(self):
        if traci.isLoaded():
            traci.close(False)
    def _init_tls(self):
        self.tls_ids = list(traci.trafficlight.getIDList())
        self.green_phases = {}
        self.yellow_after_green = {}
        self.controlled_lanes = {}
        self.current_green = {}
        max_g = 0
        for tls in self.tls_ids:
            logic = traci.trafficlight.getAllProgramLogics(tls)[0]
            phases = logic.phases
            greens, yellows = [], {}
            last_green = None
            for idx, ph in enumerate(phases):
                state = ph.state.lower()
                if "g" in state and "y" not in state:
                    greens.append(idx)
                    last_green = idx
                elif "y" in state and last_green is not None:
                    yellows[last_green] = idx
            if not greens:
                greens = [0]
            self.green_phases[tls] = greens
            self.yellow_after_green[tls] = yellows
            self.current_green[tls] = greens[0]
            traci.trafficlight.setPhase(tls, greens[0])
            lanes = []
            for ln in traci.trafficlight.getControlledLanes(tls):
                if ln not in lanes:
                    lanes.append(ln)
            self.controlled_lanes[tls] = lanes[:self.cfg.max_controlled_lanes]
            max_g = max(max_g, len(greens))
        self.action_dim = max(2, max_g)
        self.state_dim = self.cfg.max_controlled_lanes * 4 + self.action_dim
    def _is_done(self):
        return traci.simulation.getTime() >= self.cfg.max_steps or traci.simulation.getMinExpectedNumber() <= 0
    def _one_step_collect(self):
        traci.simulationStep()
        t = traci.simulation.getTime()
        for v in traci.simulation.getDepartedIDList():
            self.vehicle_depart[v] = t
        for v in traci.simulation.getArrivedIDList():
            self.metrics["throughput"] += 1
            if v in self.vehicle_depart:
                self.travel_times.append(t - self.vehicle_depart.pop(v))
        vehs = traci.vehicle.getIDList()
        if vehs:
            waits = [traci.vehicle.getWaitingTime(v) for v in vehs]
            speeds = [traci.vehicle.getSpeed(v) for v in vehs]
            fuels = [traci.vehicle.getFuelConsumption(v) for v in vehs]
            self.metrics["sum_avg_waiting_time"] += float(np.mean(waits))
            self.metrics["sum_avg_speed"] += float(np.mean(speeds))
            self.metrics["sum_fuel_mg"] += float(np.sum(fuels))
        self.metrics["sum_queue_length"] += sum(traci.lane.getLastStepHaltingNumber(l) for tls in self.tls_ids for l in self.controlled_lanes[tls])
        self.metrics["steps"] += 1
    def get_state(self, tls):
        lanes = self.controlled_lanes[tls]
        feat = []
        for ln in lanes[:self.cfg.max_controlled_lanes]:
            feat.extend([
                min(traci.lane.getLastStepVehicleNumber(ln) / self.cfg.max_lane_vehicles, 1.0),
                min(traci.lane.getLastStepHaltingNumber(ln) / self.cfg.max_lane_vehicles, 1.0),
                min(max(traci.lane.getLastStepMeanSpeed(ln), 0.0) / self.cfg.max_speed, 1.0),
                min(traci.lane.getWaitingTime(ln) / self.cfg.max_lane_waiting, 1.0),
            ])
        while len(feat) < self.cfg.max_controlled_lanes * 4:
            feat.append(0.0)
        one = [0.0] * self.action_dim
        greens = self.green_phases[tls]
        try:
            idx = greens.index(self.current_green[tls])
            one[idx] = 1.0
        except Exception:
            one[0] = 1.0
        return np.array(feat + one, dtype=np.float32)
    def get_all_states(self):
        return {tls: self.get_state(tls) for tls in self.tls_ids}
    def _apply_actions(self, actions):
        changed = []
        for tls, a in actions.items():
            greens = self.green_phases[tls]
            target = greens[a % len(greens)]
            if target != self.current_green[tls]:
                yellow = self.yellow_after_green[tls].get(self.current_green[tls])
                if yellow is not None:
                    traci.trafficlight.setPhase(tls, yellow)
                    changed.append((tls, target))
                else:
                    self.current_green[tls] = target
                    traci.trafficlight.setPhase(tls, target)
        if changed:
            for _ in range(self.cfg.yellow_duration):
                if self._is_done():
                    break
                self._one_step_collect()
            for tls, target in changed:
                self.current_green[tls] = target
                traci.trafficlight.setPhase(tls, target)
    def _reward_parts(self, tls, arrived_interval, fuel_interval):
        lanes = self.controlled_lanes[tls]
        queue = sum(traci.lane.getLastStepHaltingNumber(l) for l in lanes)
        wait = np.mean([traci.lane.getWaitingTime(l) for l in lanes]) if lanes else 0.0
        r_wait = -min(wait / self.cfg.wait_norm, 2.0)
        r_queue = -min(queue / self.cfg.queue_norm, 2.0)
        r_throughput = min((arrived_interval / max(1, len(self.tls_ids))) / self.cfg.throughput_norm, 2.0)
        r_fuel = -min((fuel_interval / max(1, len(self.tls_ids))) / self.cfg.fuel_norm, 2.0)
        return {"r_wait": r_wait, "r_queue": r_queue, "r_throughput": r_throughput, "r_fuel": r_fuel}
    def _calc_reward(self, parts, stage):
        if self.variant == "ablation_no_cl":
            return 0.35*parts["r_wait"] + 0.25*parts["r_queue"] + 0.25*parts["r_throughput"] + 0.15*parts["r_fuel"]
        if self.variant == "ablation_single_objective":
            return 0.60*parts["r_wait"] + 0.40*parts["r_queue"]
        if stage == 1:
            return 0.60*parts["r_wait"] + 0.40*parts["r_queue"]
        if stage == 2:
            return 0.45*parts["r_wait"] + 0.30*parts["r_queue"] + 0.25*parts["r_throughput"]
        return 0.35*parts["r_wait"] + 0.25*parts["r_queue"] + 0.25*parts["r_throughput"] + 0.15*parts["r_fuel"]
    def step(self, actions, stage):
        old_arrived = self.metrics["throughput"]
        old_fuel = self.metrics["sum_fuel_mg"]
        self._apply_actions(actions)
        for _ in range(self.cfg.decision_interval):
            if self._is_done():
                break
            self._one_step_collect()
        arrived_interval = self.metrics["throughput"] - old_arrived
        fuel_interval = self.metrics["sum_fuel_mg"] - old_fuel
        rewards = {}
        for tls in self.tls_ids:
            parts = self._reward_parts(tls, arrived_interval, fuel_interval)
            rewards[tls] = float(self._calc_reward(parts, stage))
            self.metrics["cumulative_reward"] += rewards[tls]
        done = self._is_done()
        next_states = self.get_all_states() if not done else {tls: np.zeros(self.state_dim, dtype=np.float32) for tls in self.tls_ids}
        return next_states, rewards, done, {}
    def summary(self):
        steps = max(1, self.metrics["steps"])
        return {
            "avg_waiting_time": self.metrics["sum_avg_waiting_time"] / steps,
            "avg_queue_length": self.metrics["sum_queue_length"] / steps,
            "throughput": self.metrics["throughput"],
            "fuel_consumption_liter_est": self.metrics["sum_fuel_mg"] / 748900.0,
            "avg_speed": self.metrics["sum_avg_speed"] / steps,
            "avg_travel_time": float(np.mean(self.travel_times)) if self.travel_times else 0.0,
            "cumulative_reward": self.metrics["cumulative_reward"],
            "avg_loss": self.metrics["loss_sum"] / max(1, self.metrics["loss_count"]),
            "steps": steps,
        }


def stage_of_episode(ep, cfg, variant):
    if variant == "ablation_no_cl":
        return 3
    if ep <= cfg.stage1_end:
        return 1
    if ep <= cfg.stage2_end:
        return 2
    return 3


def append_csv(path, row):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def train(cfg, variant, gui=False):
    set_seed(cfg.seed)
    env = MultiIntersectionEnv(cfg, variant, gui)
    states = env.reset(seed=cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agent = D3QNAgent(env.state_dim, env.action_dim, cfg, device)
    env.close()

    out = Path(cfg.output_dir) / variant
    csv_path = out / "training_metrics.csv"
    best = -float("inf")

    print(f"Konteks: Indonesia, left-hand traffic, right-hand steering")
    print(f"Variant: {variant} | Device: {device}")
    print(f"State dim: {env.state_dim} | Action dim: {env.action_dim} | TLS agents: {len(env.tls_ids)}")

    try:
        for ep in range(1, cfg.episodes + 1):
            stage = stage_of_episode(ep, cfg, variant)
            states = env.reset(seed=cfg.seed + ep)
            done = False
            while not done:
                actions = {tls: agent.select_action(s) for tls, s in states.items()}
                next_states, rewards, done, _ = env.step(actions, stage)
                for tls in env.tls_ids:
                    agent.replay.push(states[tls], actions[tls], rewards[tls], next_states[tls], done)
                    loss = agent.optimize()
                    if loss is not None:
                        env.metrics["loss_sum"] += loss
                        env.metrics["loss_count"] += 1
                states = next_states
            summ = env.summary()
            row = {"variant": variant, "episode": ep, "stage": stage, "epsilon": agent.epsilon(), **summ}
            append_csv(csv_path, row)
            if summ["cumulative_reward"] > best:
                best = summ["cumulative_reward"]
                agent.save(out / "best_model.pt")
            agent.save(out / "last_model.pt")
            print(f"{variant} | EP {ep:03d} | Stage {stage} | R {summ['cumulative_reward']:.3f} | AWT {summ['avg_waiting_time']:.2f}s | AQL {summ['avg_queue_length']:.2f} | TP {summ['throughput']} | Fuel {summ['fuel_consumption_liter_est']:.3f}L | Speed {summ['avg_speed']:.2f} | ATT {summ['avg_travel_time']:.2f}s | Loss {summ['avg_loss']:.5f} | eps {agent.epsilon():.3f}")
    finally:
        env.close()
    print(f"Selesai. Log: {csv_path}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--sumocfg", default="city1_indonesia_2lane_leftfree.sumocfg")
    p.add_argument("--variant", default="full_mo_cl_d3qn", choices=["full_mo_cl_d3qn", "ablation_no_cl", "ablation_single_objective"])
    p.add_argument("--episodes", type=int, default=120)
    p.add_argument("--max-steps", type=int, default=3600)
    p.add_argument("--gui", action="store_true")
    p.add_argument("--output-dir", default="outputs_multi_indonesia")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--stage1-end", type=int, default=40)
    p.add_argument("--stage2-end", type=int, default=80)
    p.add_argument("--decision-interval", type=int, default=10)
    p.add_argument("--yellow-duration", type=int, default=3)
    return p.parse_args()


def main():
    a = parse_args()
    cfg = Config(sumocfg=a.sumocfg, output_dir=a.output_dir, episodes=a.episodes,
                 max_steps=a.max_steps, seed=a.seed, stage1_end=a.stage1_end,
                 stage2_end=a.stage2_end, decision_interval=a.decision_interval,
                 yellow_duration=a.yellow_duration)
    train(cfg, a.variant, gui=a.gui)


if __name__ == "__main__":
    main()
