"""
Worker Agent 1 — Data Quality Triage Agent.
Round 1 Component.
"""

from __future__ import annotations
import random
from workers.base_worker import BaseWorker

TRIAGE_TASK_CONFIGS = {
    "easy_triage": {"step_budget": 8},
    "medium_triage": {"step_budget": 10},
    "hard_triage": {"step_budget": 12},
}

class DataQualityTriageEnv(BaseWorker):
    """
    Worker Agent 1: Data Quality Triage.
    """
    VALID_ACTIONS = ["inspect_schema", "clean_nulls", "deduplicate", "validate_types", "submit"]

    def __init__(self, task_id: str = "easy_triage") -> None:
        super().__init__(worker_id="worker_1", worker_name="Data Quality Triage")
        self.task_id = task_id
        self.config = TRIAGE_TASK_CONFIGS.get(task_id, TRIAGE_TASK_CONFIGS["easy_triage"])
        self.step_budget = self.config["step_budget"]
        self.step_budget_remaining = self.step_budget

    def reset(self, task_id: str) -> dict:
        self._reset_episode_tracking()
        self.task_id = task_id
        self.config = TRIAGE_TASK_CONFIGS.get(task_id, TRIAGE_TASK_CONFIGS["easy_triage"])
        self.step_budget = self.config["step_budget"]
        self.step_budget_remaining = self.step_budget
        return self.state()

    def step(self, action_dict: dict) -> tuple[dict, float, bool, dict]:
        operation = action_dict.get("operation", "unknown")
        reward = 0.2
        if operation == "submit":
            self.submitted = True
            self.is_done = True
            reward = 0.5
        
        self.step_count += 1
        self.step_budget_remaining -= 1
        self.total_reward += reward
        done = self.is_done or self._is_budget_exhausted()
        info = {"action": operation}
        self._record_action(operation, reward, info)
        return self.state(), reward, done, info

    def state(self) -> dict:
        return self._get_base_state()

    def generate_run_report(self) -> dict:
        return self._get_base_report()

    def evaluate_run(self) -> dict:
        score = min(max(self.total_reward / 5.0, 0.0), 1.0)
        return {"approved": score >= 0.5, "gates": {}, "composite_score": score}
