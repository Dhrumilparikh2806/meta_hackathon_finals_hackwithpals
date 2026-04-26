"""
Abstract base class for all Fleet worker agents.
Enforces OpenEnv reset/step/state contract on every worker.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    pass


class BaseWorker(ABC):
    """
    Abstract base class for all worker agents in the Fleet AI Oversight Environment.
    
    Every worker must implement:
    - reset(task_id: str) -> dict
    - step(action_dict: dict) -> tuple[dict, float, bool, dict]
    - state() -> dict
    - generate_run_report() -> dict
    - evaluate_run() -> dict
    """

    # ------------------------------------------------------------------ #
    # Constructor                                                           #
    # ------------------------------------------------------------------ #

    def __init__(self, worker_id: str, worker_name: str) -> None:
        self.worker_id: str = worker_id
        self.worker_name: str = worker_name

        # Episode tracking
        self.step_count: int = 0
        self.total_reward: float = 0.0
        self.action_history: list[dict] = []
        self.governance_events: list[dict] = []
        self.episode_start_time: Optional[float] = None
        self.task_id: Optional[str] = None

        # Budget tracking
        self.step_budget: int = 0
        self.step_budget_remaining: int = 0

        # Status
        self.is_done: bool = False
        self.submitted: bool = False

    # ------------------------------------------------------------------ #
    # Abstract Methods — must be implemented by every worker               #
    # ------------------------------------------------------------------ #

    @abstractmethod
    def reset(self, task_id: str) -> dict:
        """
        Reset environment to initial state for given task_id.
        Returns initial observation dict.
        Must set self.task_id, self.step_budget, self.step_budget_remaining.
        Must reset all episode tracking fields.
        """
        raise NotImplementedError

    @abstractmethod
    def step(self, action_dict: dict) -> tuple[dict, float, bool, dict]:
        """
        Apply action to environment.
        Returns (observation, reward, done, info).
        reward must always be in [0.0, 1.0].
        done=True when budget exhausted or submit action called.
        info dict must contain at minimum: {"error": None, "action": str}
        """
        raise NotImplementedError

    @abstractmethod
    def state(self) -> dict:
        """
        Return current full internal state.
        Must include: worker_id, worker_name, task_id, step_count,
        step_budget_remaining, total_reward, is_done, submitted.
        """
        raise NotImplementedError

    @abstractmethod
    def generate_run_report(self) -> dict:
        """
        Generate full run report at episode end.
        Must include: worker_id, task_id, step_count, total_reward,
        action_history, governance_events, submitted, final_score.
        """
        raise NotImplementedError

    @abstractmethod
    def evaluate_run(self) -> dict:
        """
        Gate-based evaluation of the run.
        Must return: {"approved": bool, "gates": dict, "composite_score": float}
        composite_score must be in (0.0, 1.0) via epsilon clipping.
        """
        raise NotImplementedError

    # ------------------------------------------------------------------ #
    # Concrete Helper Methods — available to all workers                   #
    # ------------------------------------------------------------------ #

    def _record_governance_event(
        self,
        event_type: str,
        severity: str,
        detail: str,
    ) -> None:
        """
        Record a governance event in the episode log.
        severity must be one of: 'low', 'medium', 'high', 'critical'
        """
        assert severity in ("low", "medium", "high", "critical"), \
            f"Invalid severity: {severity}"
        
        event = {
            "step": self.step_count,
            "event_type": event_type,
            "severity": severity,
            "detail": detail,
            "timestamp": time.time(),
        }
        self.governance_events.append(event)

    def _is_budget_exhausted(self) -> bool:
        """Returns True if step budget is fully consumed."""
        return self.step_budget_remaining <= 0

    def _clip_reward(self, r: float, lo: float = 0.0, hi: float = 0.99) -> float:
        """
        Clip reward to [lo, hi] range.
        Default range is [0.0, 0.99] — never exactly 1.0 until submit.
        Uses epsilon to avoid exact boundary values.
        """
        epsilon = 1e-6
        return float(max(lo + epsilon, min(hi - epsilon, r)))

    def _record_action(self, action_name: str, reward: float, info: dict) -> None:
        """Record action in episode history."""
        self.action_history.append({
            "step": self.step_count,
            "action": action_name,
            "reward": reward,
            "info": info,
            "timestamp": time.time(),
        })

    def _reset_episode_tracking(self) -> None:
        """Reset all episode tracking fields. Call at start of reset()."""
        self.step_count = 0
        self.total_reward = 0.0
        self.action_history = []
        self.governance_events = []
        self.episode_start_time = time.time()
        self.is_done = False
        self.submitted = False

    def _get_base_state(self) -> dict:
        """Returns base state fields common to all workers."""
        return {
            "worker_id": self.worker_id,
            "worker_name": self.worker_name,
            "task_id": self.task_id,
            "step_count": self.step_count,
            "step_budget_remaining": self.step_budget_remaining,
            "total_reward": round(self.total_reward, 4),
            "is_done": self.is_done,
            "submitted": self.submitted,
            "governance_event_count": len(self.governance_events),
        }

    def _get_base_report(self) -> dict:
        """Returns base report fields common to all workers."""
        elapsed = (
            round(time.time() - self.episode_start_time, 2)
            if self.episode_start_time
            else 0.0
        )
        return {
            "worker_id": self.worker_id,
            "worker_name": self.worker_name,
            "task_id": self.task_id,
            "step_count": self.step_count,
            "step_budget": self.step_budget,
            "steps_used": self.step_count,
            "budget_utilization": round(
                self.step_count / max(self.step_budget, 1), 4
            ),
            "total_reward": round(self.total_reward, 4),
            "avg_reward_per_step": round(
                self.total_reward / max(self.step_count, 1), 4
            ),
            "action_history": self.action_history,
            "governance_events": self.governance_events,
            "submitted": self.submitted,
            "elapsed_seconds": elapsed,
        }
