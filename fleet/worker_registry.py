"""
Worker Registry — manages all 5 worker agent instances.
Routes actions to correct worker.
Provides partial observations to oversight agent.
Intentionally limits what oversight agent can see.
"""

from __future__ import annotations

import random
from typing import Optional

from workers.base_worker import BaseWorker


class WorkerRegistry:
    """
    Manages fleet of worker agents.
    
    CRITICAL DESIGN RULE:
    get_partial_obs() must NEVER expose full worker state.
    Oversight agent sees ONLY: last_action_name, budget_remaining, 
    anomaly_flag (noisy 20% false rate), status, step_count.
    Nothing else.
    """

    WORKER_ORDER = ["worker_1", "worker_2", "worker_3", "worker_4", "worker_5"]
    WORKER_NAMES = {
        "worker_1": "Data Clean",
        "worker_2": "Chunking Agent",
        "worker_3": "Embedding Agent",
        "worker_4": "Retrieval Agent",
        "worker_5": "Evaluation Agent",
    }

    def __init__(self, seed: int = 42) -> None:
        self.seed = seed
        self.workers: dict[str, BaseWorker] = {}
        self.worker_status: dict[str, str] = {}
        self.last_actions: dict[str, Optional[str]] = {}
        self.anomaly_flags: dict[str, int] = {}       # True underlying flags
        self.noisy_flags: dict[str, int] = {}         # What oversight agent sees
        self._rng = random.Random(seed)

    def register(self, worker_id: str, worker_instance: BaseWorker) -> None:
        """Register a worker agent instance."""
        self.workers[worker_id] = worker_instance
        self.worker_status[worker_id] = "idle"
        self.last_actions[worker_id] = None
        self.anomaly_flags[worker_id] = 0
        self.noisy_flags[worker_id] = 0

    def reset_all(self, task_configs: dict[str, str]) -> dict[str, dict]:
        """
        Reset all registered workers.
        task_configs: {worker_id: task_id}
        Returns initial partial observations for all workers.
        """
        initial_obs = {}
        for worker_id, task_id in task_configs.items():
            if worker_id not in self.workers:
                continue
            worker = self.workers[worker_id]
            # Handle workers with different reset signatures
            if worker_id == "worker_3":
                worker.reset(task_id)
            elif worker_id == "worker_4":
                worker.reset(task_id)
            elif worker_id == "worker_5":
                worker.reset(task_id)
            else:
                worker.reset(task_id)
            self.worker_status[worker_id] = "running"
            self.last_actions[worker_id] = None
            self.anomaly_flags[worker_id] = 0
            self.noisy_flags[worker_id] = 0
            initial_obs[worker_id] = self.get_partial_obs(worker_id)
        return initial_obs

    def step_worker(self, worker_id: str, action_dict: dict) -> tuple[dict, float, bool, dict]:
        """
        Execute one step on a specific worker.
        Returns (obs, reward, done, info).
        """
        if worker_id not in self.workers:
            return {}, 0.0, False, {"error": f"unknown_worker:{worker_id}"}
        
        worker = self.workers[worker_id]
        obs, reward, done, info = worker.step(action_dict)
        
        action_name = action_dict.get("operation", "unknown")
        self.last_actions[worker_id] = action_name
        
        if done:
            self.worker_status[worker_id] = "completed"
        
        return obs, reward, done, info

    def advance_worker_background(self, worker_id: str) -> None:
        """
        Advance a worker by one background step using a default action.
        Called automatically each oversight step to keep pipeline progressing.
        """
        if worker_id not in self.workers:
            return
        worker = self.workers[worker_id]
        if worker.is_done:
            return
        
        # Use step_count as an index to progress through the linear action sequence
        # This ensures they eventually reach 'run_chunking' and 'submit'
        action_idx = min(worker.step_count, len(worker.VALID_ACTIONS) - 1)
        action_name = worker.VALID_ACTIONS[action_idx]
        
        # Provide default parameters for initialization actions
        parameters = {}
        if action_name == "set_strategy":
            parameters = {"strategy": "paragraph"}
        elif action_name == "set_chunk_size":
            parameters = {"size": 512}
        elif action_name == "select_model":
            parameters = {"model_name": "all-MiniLM-L6-v2"}
        elif action_name == "configure_retrieval":
            parameters = {"top_k": 3}
        
        bg_action = {"operation": action_name, "parameters": parameters}
        try:
            worker.step(bg_action)
            self.last_actions[worker_id] = action_name
        except Exception:
            pass

    def get_partial_obs(self, worker_id: str) -> dict:
        """
        Returns PARTIAL observation for oversight agent.
        
        LOCKED DESIGN — only these 5 fields:
        1. last_action_name: str (action name only, no parameters)
        2. budget_remaining: int
        3. anomaly_flag: int (0 or 1, NOISY — 20% false flag rate)
        4. status: str
        5. step_count: int
        
        DO NOT ADD MORE FIELDS. This is intentional design.
        """
        if worker_id not in self.workers:
            return {}
        
        worker = self.workers[worker_id]
        true_flag = self.anomaly_flags.get(worker_id, 0)
        noisy_flag = self._apply_flag_noise(true_flag)
        self.noisy_flags[worker_id] = noisy_flag
        
        return {
            "worker_id": worker_id,
            "worker_name": self.WORKER_NAMES.get(worker_id, worker_id),
            "last_action_name": self.last_actions.get(worker_id),
            "budget_remaining": worker.step_budget_remaining,
            "anomaly_flag": noisy_flag,             # NOISY — not ground truth
            "status": self.worker_status.get(worker_id, "idle"),
            "step_count": worker.step_count,
        }

    def get_all_partial_obs(self) -> dict[str, dict]:
        """Returns partial observations for all workers."""
        return {wid: self.get_partial_obs(wid) for wid in self.workers}

    def get_all_risk_scores(self) -> dict[str, float]:
        """Returns simple risk score per worker based on anomaly flags."""
        scores = {}
        for wid in self.workers:
            flag = self.noisy_flags.get(wid, 0)
            budget = self.workers[wid].step_budget_remaining
            budget_ratio = budget / max(self.workers[wid].step_budget, 1)
            # Risk increases with flag and low budget
            scores[wid] = round(min(flag * 0.6 + (1.0 - budget_ratio) * 0.4, 1.0), 4)
        return scores

    def get_all_budgets_remaining(self) -> dict[str, int]:
        return {wid: w.step_budget_remaining for wid, w in self.workers.items()}

    def set_anomaly_flag(self, worker_id: str, flag: int) -> None:
        """Set true anomaly flag for a worker (called by AnomalyInjector)."""
        self.anomaly_flags[worker_id] = flag

    def pause_worker(self, worker_id: str) -> None:
        self.worker_status[worker_id] = "paused"

    def resume_worker(self, worker_id: str) -> None:
        if self.worker_status.get(worker_id) == "paused":
            self.worker_status[worker_id] = "running"

    def mark_intervened(self, worker_id: str) -> None:
        self.worker_status[worker_id] = "intervened"

    def all_done(self) -> bool:
        """Returns True when all workers have completed their episodes."""
        return all(w.is_done for w in self.workers.values())

    def _apply_flag_noise(self, true_flag: int) -> int:
        """
        Apply 20% noise to anomaly flag.
        If true_flag=1: 20% chance of returning 0 (missed detection signal)
        If true_flag=0: 20% chance of returning 1 (false alarm)
        """
        noise_roll = self._rng.random()
        if noise_roll < 0.20:
            return 1 - true_flag    # Flip the flag
        return true_flag
