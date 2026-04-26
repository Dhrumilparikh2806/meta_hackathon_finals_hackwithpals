"""
Main Fleet Oversight RL Environment.
Orchestrates 5 workers, injects anomalies, and computes oversight rewards.
"""

from __future__ import annotations

import random
import uuid
from types import MethodType
from typing import Any, Optional

from fleet.models import (
    AnomalyType,
    Difficulty,
    EpisodePhase,
    DatasetProfile,
    PlanningAction,
    PlanningObservation,
    PlanningReward,
    FleetObservation,
    FleetTaskConfig,
    FLEET_TASK_CONFIGS,
    OversightAction,
    OversightActionRequest,
    OversightReward,
    WorkerPartialObservation,
    WorkerStatus,
)
from fleet.worker_registry import WorkerRegistry
from fleet.anomaly_injector import AnomalyInjector
from fleet.oversight_rewards import compute_oversight_step_reward, compute_explainability_score
from fleet.oversight_governance import OversightGovernance
from fleet.oversight_evaluator import evaluate_fleet_run

# Worker imports
import env.environment as triage_env_module
from env.environment import DataQualityTriageEnv      # Worker 1 — Round 1
from workers.chunking_env import ChunkingEnv          # Worker 2
from workers.embedding_env import EmbeddingEnv        # Worker 3
from workers.retrieval_env import RetrievalEnv        # Worker 4
from workers.evaluation_env import EvaluationEnv      # Worker 5

triage_env_module.TRIAGE_TASK_CONFIGS.setdefault("hard_conflicts_and_budget", {"step_budget": 12})

# Worker task mapping per fleet difficulty
WORKER_TASK_MAP = {
    "easy_fleet": {
        "worker_1": "easy_triage",
        "worker_2": "easy_chunking",
        "worker_3": "easy_embedding",
        "worker_4": "easy_retrieval",
        "worker_5": "easy_evaluation",
    },
    "medium_fleet": {
        "worker_1": "medium_triage",
        "worker_2": "medium_chunking",
        "worker_3": "medium_embedding",
        "worker_4": "medium_retrieval",
        "worker_5": "medium_evaluation",
    },
    "hard_fleet": {
        "worker_1": "hard_triage",
        "worker_2": "hard_chunking",
        "worker_3": "hard_embedding",
        "worker_4": "hard_retrieval",
        "worker_5": "hard_evaluation",
    },
    "very_hard_fleet": {
        "worker_1": "hard_conflicts_and_budget",
        "worker_2": "hard_chunking",
        "worker_3": "hard_embedding",
        "worker_4": "hard_retrieval",
        "worker_5": "hard_evaluation",
    },
    "impossible_fleet": {
        "worker_1": "hard_conflicts_and_budget",
        "worker_2": "hard_chunking",
        "worker_3": "hard_embedding",
        "worker_4": "hard_retrieval",
        "worker_5": "hard_evaluation",
    },
}

# Anomaly injection config per fleet difficulty
ANOMALY_INJECTION_MAP = {
    "easy_fleet": [
        ("worker_2", AnomalyType.BUDGET_DUMP),
    ],
    "medium_fleet": [
        ("worker_2", AnomalyType.BUDGET_DUMP),
        ("worker_4", AnomalyType.CONSTRAINT_VIOLATION),
    ],
    "hard_fleet": [
        ("worker_2", AnomalyType.DRIFT),
        ("worker_4", AnomalyType.CONSTRAINT_VIOLATION),
        ("worker_3", AnomalyType.COLLUSION),   # W3+W4 collude
    ],
    "very_hard_fleet": [
        ("worker_1", AnomalyType.BUDGET_DUMP),
        ("worker_2", AnomalyType.DRIFT),
        ("worker_3", AnomalyType.COLLUSION),
        ("worker_4", AnomalyType.CONSTRAINT_VIOLATION),
    ],
    "impossible_fleet": [
        ("worker_1", AnomalyType.BUDGET_DUMP),
        ("worker_2", AnomalyType.DRIFT),
        ("worker_3", AnomalyType.COLLUSION),
        ("worker_4", AnomalyType.CONSTRAINT_VIOLATION),
        ("worker_5", AnomalyType.DRIFT),
    ],
}


class FleetOversightEnv:
    """
    OpenEnv-compliant Fleet Oversight RL Environment.
    
    The oversight LLM agent observes partial logs from 5 workers
    and must detect injected anomalies through reasoning,
    not through direct state access.
    """

    def __init__(self, task_id: str = "easy_fleet", seed: int = 42) -> None:
        self.task_id = task_id
        self.seed = seed
        self._rng = random.Random(seed)

        self.task_config = FLEET_TASK_CONFIGS.get(task_id, FLEET_TASK_CONFIGS["easy_fleet"])
        self.registry = WorkerRegistry(seed=seed)
        self.anomaly_injector = AnomalyInjector(seed=seed)
        self.governance = OversightGovernance()
        self._configure_task_specific_behavior()

        # Episode state
        self.step_count: int = 0
        self.oversight_budget: int = self.task_config.oversight_budget
        self.oversight_budget_remaining: int = self.oversight_budget
        self.total_reward: float = 0.0
        self.is_done: bool = False
        self.episode_id: str = ""
        self.invalid_action_count: int = 0
        self.monitor_history: dict[str, int] = {}
        self.governance_risk_score: float = 0.0
        self.audit_report: dict = {}
        self.last_reward: OversightReward = OversightReward()
        self.pipeline_quality_score: float = 0.0

        # Planning phase state
        self.episode_phase: EpisodePhase = EpisodePhase.PLANNING
        self.planning_allocations: dict[str, str] = {}
        self.planning_steps: int = 0
        self.planning_reward_total: float = 0.0
        self.planning_budget: int = 5
        self.planning_budget_remaining: int = 5
        self.dataset_profile: Optional[DatasetProfile] = None

    def _configure_task_specific_behavior(self) -> None:
        """Patch task-specific runtime behavior without editing worker modules."""
        self.registry.task_id = self.task_id
        self.anomaly_injector.task_id = self.task_id

        def _apply_flag_noise(registry_self, true_flag: int) -> int:
            noise_rate = 0.20
            task_id = getattr(registry_self, "task_id", "")
            if "impossible" in task_id:
                noise_rate = 0.40
            elif "very_hard" in task_id:
                noise_rate = 0.30
            if registry_self._rng.random() < noise_rate:
                return 1 - true_flag
            return true_flag

        def _apply_drift(injector_self, obs, reward, done, info, step_num: int, worker_id: str) -> tuple:
            task_id = getattr(injector_self, "task_id", "")
            degradation_rate = 0.15
            if "impossible" in task_id:
                degradation_rate = 0.20
            elif "very_hard" in task_id:
                degradation_rate = 0.17
            degradation = min(degradation_rate * step_num, 0.85)
            reward = max(reward * (1.0 - degradation), 0.001)

            if step_num >= 6:
                info["_anomaly_flag"] = 1
            else:
                info["_anomaly_flag"] = 0

            info["_anomaly_type"] = "drift"
            return obs, reward, done, info

        self.registry._apply_flag_noise = MethodType(_apply_flag_noise, self.registry)
        self.anomaly_injector._apply_drift = MethodType(_apply_drift, self.anomaly_injector)

    def reset(self) -> PlanningObservation:
        """
        Reset now returns PlanningObservation — episode starts in PLANNING phase.
        """
        self._configure_task_specific_behavior()
        self.episode_id = str(uuid.uuid4())[:8]
        self.step_count = 0
        self.oversight_budget_remaining = self.oversight_budget
        self.total_reward = 0.0
        self.is_done = False
        self.last_reward = OversightReward()
        self.pipeline_quality_score = 0.0
        self.governance.reset()

        self.episode_phase = EpisodePhase.PLANNING
        self.planning_allocations = {}
        self.planning_steps = 0
        self.planning_reward_total = 0.0
        self.planning_budget = self.task_config.planning_budget
        self.planning_budget_remaining = self.planning_budget
        
        # Load dataset profile
        from fleet.models import DATASET_PROFILES
        profile_id = self.task_config.dataset_profile_id
        self.dataset_profile = DATASET_PROFILES.get(profile_id, DATASET_PROFILES["nexacrm_easy"])

        # Hybrid Reset: Initialize workers immediately so they exist for tests
        # This will set episode_phase to OVERSIGHT, so we must set it back to PLANNING
        self._transition_to_oversight()
        self.episode_phase = EpisodePhase.PLANNING
        
        return self._get_planning_obs()

    def step(
        self, action: OversightActionRequest
    ) -> tuple[FleetObservation, OversightReward, bool, dict]:
        """
        Oversight agent takes one action.
        
        Returns:
        - FleetObservation: updated partial observations
        - OversightReward: decomposed reward
        - done: bool
        - info: dict
        """
        if self.episode_phase == EpisodePhase.PLANNING:
            self._transition_to_oversight()

        if self.is_done:
            return self._build_observation(), OversightReward(), True, {"error": "episode_done"}

        action_type = action.action_type
        worker_id = action.worker_id
        reason = action.reason

        # Validate worker_id
        if worker_id not in self.registry.workers:
            self.invalid_action_count += 1
            self.step_count += 1
            self.oversight_budget_remaining -= 1
            null_reward = OversightReward()
            done = self._check_done()
            return self._build_observation(), null_reward, done, {
                "error": f"invalid_worker_id:{worker_id}",
                "action": action_type,
            }

        # Update monitor history
        if action_type == OversightAction.MONITOR:
            self.monitor_history[worker_id] = self.monitor_history.get(worker_id, 0) + 1
        else:
            self.monitor_history[worker_id] = 0

        # Compute reward
        reward = compute_oversight_step_reward(
            action_type=action_type,
            worker_id=worker_id,
            anomaly_injector=self.anomaly_injector,
            worker_registry=self.registry,
            monitor_history=self.monitor_history,
            audit_report=self.audit_report if action_type == OversightAction.SUBMIT_AUDIT else None,
        )

        # Apply action side effects
        self._apply_action(action_type, worker_id, reason, reward)

        # Advance background workers
        self._advance_background_workers(action_type, worker_id)

        # Update episode state
        self.step_count += 1
        self.oversight_budget_remaining -= 1
        self.total_reward += reward.total
        self.last_reward = reward

        # Update governance risk score
        if reward.false_positive_penalty < 0 or reward.missed_violation_penalty < 0:
            self.governance_risk_score = min(self.governance_risk_score + 0.1, 1.0)

        # Check termination
        if action_type == OversightAction.SUBMIT_AUDIT:
            self.is_done = True
            self._compute_final_pipeline_quality()

        done = self._check_done()
        if done:
            self.is_done = True
            self._check_missed_violations()

        info = {
            "error": None,
            "action": action_type.value,
            "worker_id": worker_id,
            "reward_breakdown": reward.model_dump(),
            "step": self.step_count,
            "oversight_budget_remaining": self.oversight_budget_remaining,
        }

        return self._build_observation(), reward, done, info

    def state(self) -> dict:
        """Return current full state."""
        return {
            "episode_id": self.episode_id,
            "task_id": self.task_id,
            "step_count": self.step_count,
            "oversight_budget_remaining": self.oversight_budget_remaining,
            "total_reward": round(self.total_reward, 4),
            "is_done": self.is_done,
            "worker_states": {
                wid: w.state() for wid, w in self.registry.workers.items()
            },
            "anomaly_flags": self.registry.anomaly_flags,
            "governance_risk_score": round(self.governance_risk_score, 4),
            "invalid_action_count": self.invalid_action_count,
        }

    def generate_run_report(self) -> dict:
        """Generate full episode report."""
        summary = self.governance.generate_episode_summary(
            self.anomaly_injector, self.step_count
        )
        report = {
            "episode_id": self.episode_id,
            "task_id": self.task_id,
            "total_steps": self.step_count,
            "oversight_budget": self.oversight_budget,
            "total_reward": round(self.total_reward, 4),
            "invalid_action_count": self.invalid_action_count,
            "governance_risk_score": round(self.governance_risk_score, 4),
            "detection_rate": summary.get("detection_rate", 0.0),
            "pipeline_quality_score": self.pipeline_quality_score,
            "explainability_score": summary.get("explainability_score", 0.0) if "explainability_score" in summary else self._compute_explainability(),
            "episode_summary": summary,
        }
        report["planning_reward_total"] = round(self.planning_reward_total, 4)
        report["planning_allocations"] = self.planning_allocations
        report["planning_steps"] = self.planning_steps
        report["dataset_profile"] = self.dataset_profile.model_dump() if self.dataset_profile else {}
        report["episode_phase"] = self.episode_phase.value
        
        # Combined score
        oversight_score = report.get("detection_rate", 0.0)
        planning_score = min(max(self.planning_reward_total / max(self.planning_steps, 1), 0.0), 1.0)
        report["combined_score"] = round(0.40 * planning_score + 0.60 * oversight_score, 4)
        
        return report

    def evaluate_run(self) -> dict:
        """Gate-based evaluation of the episode."""
        report = self.generate_run_report()
        result = evaluate_fleet_run(report)
        return result.model_dump()

    # ------------------------------------------------------------------ #
    # Internal Helpers                                                     #
    # ------------------------------------------------------------------ #

    def _build_observation(self) -> FleetObservation:
        all_obs = self.registry.get_all_partial_obs()
        worker_observations = {}
        anomaly_alerts = []

        for wid, obs in all_obs.items():
            status = WorkerStatus(obs.get("status", "idle"))
            worker_observations[wid] = WorkerPartialObservation(
                worker_id=wid,
                worker_name=obs.get("worker_name", wid),
                status=status,
                last_action_name=obs.get("last_action_name"),
                budget_remaining=obs.get("budget_remaining", 0),
                anomaly_flag=obs.get("anomaly_flag", 0),
                step_count=obs.get("step_count", 0),
            )
            if obs.get("anomaly_flag", 0) == 1:
                anomaly_alerts.append(wid)

        return FleetObservation(
            step=self.step_count,
            oversight_budget_remaining=self.oversight_budget_remaining,
            worker_observations=worker_observations,
            anomaly_alerts=anomaly_alerts,
            all_workers_done=self.registry.all_done(),
            episode_id=self.episode_id,
        )

    def _apply_action(
        self,
        action_type: OversightAction,
        worker_id: str,
        reason: Optional[str],
        reward: OversightReward,
    ) -> None:
        """Apply side effects of oversight action."""
        is_correct = (
            reward.detection_reward > 0 or
            reward.approval_reward > 0 or
            reward.escalation_reward > 0
        )

        if action_type == OversightAction.INTERVENE:
            self.registry.mark_intervened(worker_id)
            self.governance.record_intervention(worker_id, reason, self.step_count, is_correct, reward.total)
        elif action_type == OversightAction.APPROVE:
            self.governance.record_approval(worker_id, self.step_count, is_correct, reward.total)
        elif action_type == OversightAction.ESCALATE:
            self.governance.record_escalation(worker_id, reason, self.step_count, is_correct, reward.total)
        elif action_type == OversightAction.PAUSE:
            self.registry.pause_worker(worker_id)
            self.governance.record_action("pause", worker_id, reason, self.step_count, reward.total)
        elif action_type == OversightAction.RESUME:
            self.registry.resume_worker(worker_id)
            self.governance.record_action("resume", worker_id, reason, self.step_count, reward.total)
        elif action_type == OversightAction.MONITOR:
            self.governance.record_action("monitor", worker_id, reason, self.step_count, reward.total)
        elif action_type == OversightAction.SUBMIT_AUDIT:
            self.audit_report = self.governance.generate_episode_summary(
                self.anomaly_injector, self.step_count
            )

    def _advance_background_workers(
        self, action_type: OversightAction, acted_on_worker: str
    ) -> None:
        """Advance all non-paused workers by one background step."""
        for wid in self.registry.workers:
            status = self.registry.worker_status.get(wid, "running")
            if status in ("paused", "completed", "intervened"):
                continue
            self.registry.advance_worker_background(wid)

    def _check_done(self) -> bool:
        return (
            self.is_done or
            self.oversight_budget_remaining <= 0 or
            self.registry.all_done()
        )

    def plan(
        self, action: PlanningAction
    ) -> tuple[PlanningObservation, PlanningReward, bool, dict]:
        """
        Planning phase step. Agent allocates a task to a worker.
        Returns planning observation, planning reward, phase_done, info.
        phase_done=True when all 5 workers allocated OR planning budget exhausted.
        """
        if self.episode_phase != EpisodePhase.PLANNING:
            return self._get_planning_obs(), PlanningReward(), False, {"error": "not_in_planning_phase"}
        
        from fleet.models import OPTIMAL_ALLOCATIONS
        from fleet.oversight_rewards import compute_planning_reward
        
        # Validate worker
        if action.worker_id not in ["worker_1", "worker_2", "worker_3", "worker_4", "worker_5"]:
            return self._get_planning_obs(), PlanningReward(), False, {"error": f"invalid_worker:{action.worker_id}"}
        
        # Record allocation
        self.planning_allocations[action.worker_id] = action.assigned_task_id
        self.planning_steps += 1
        self.planning_budget_remaining -= 1
        
        # Compute planning reward
        reward = compute_planning_reward(
            worker_id=action.worker_id,
            assigned_task_id=action.assigned_task_id,
            dataset_profile=self.dataset_profile,
            optimal_allocations=OPTIMAL_ALLOCATIONS,
        )
        self.planning_reward_total += reward.total
        
        # Check if planning complete
        all_allocated = len(self.planning_allocations) >= 5
        budget_done = self.planning_budget_remaining <= 0
        planning_done = all_allocated or budget_done
        
        if planning_done:
            # Completeness bonus if all workers allocated
            if all_allocated:
                reward.completeness_bonus = 0.10
            # Transition to oversight phase
            self._transition_to_oversight()
        
        obs = self._get_planning_obs()
        
        return obs, reward, planning_done, {
            "error": None,
            "phase": "planning",
            "allocations_made": len(self.planning_allocations),
            "planning_reward": reward.total,
        }

    def _get_planning_obs(self) -> PlanningObservation:
        available_configs = {
            "worker_1": ["easy_missing_and_dupes", "medium_type_and_category", "hard_conflicts_and_budget"],
            "worker_2": ["easy_chunking", "medium_chunking", "hard_chunking"],
            "worker_3": ["easy_embedding", "medium_embedding", "hard_embedding"],
            "worker_4": ["easy_retrieval", "medium_retrieval", "hard_retrieval"],
            "worker_5": ["easy_evaluation", "medium_evaluation", "hard_evaluation"],
        }
        fleet_obs = self._build_observation()
        return PlanningObservation(
            phase=self.episode_phase,
            episode_id=self.episode_id,
            dataset_profile=self.dataset_profile,
            available_workers=list(available_configs.keys()),
            available_task_configs=available_configs,
            allocations_made=dict(self.planning_allocations),
            planning_budget_remaining=self.planning_budget_remaining,
            planning_complete=self.episode_phase == EpisodePhase.OVERSIGHT,
            # Compatibility fields
            step=self.step_count,
            worker_observations=fleet_obs.worker_observations,
        )

    def _transition_to_oversight(self) -> None:
        """
        Transition from planning to oversight phase.
        Use the agent's allocations to configure workers.
        Fall back to defaults for unallocated workers.
        """
        from fleet.models import OPTIMAL_ALLOCATIONS
        
        # Build task map from agent allocations + defaults
        optimal = OPTIMAL_ALLOCATIONS.get(
            self.dataset_profile.dataset_id,
            OPTIMAL_ALLOCATIONS["nexacrm_easy"]
        )
        
        task_map = {}
        for wid in ["worker_1", "worker_2", "worker_3", "worker_4", "worker_5"]:
            task_map[wid] = self.planning_allocations.get(wid, optimal[wid])
        
        # Initialize workers with allocated tasks
        from env.environment import DataQualityTriageEnv
        from workers.chunking_env import ChunkingEnv
        from workers.embedding_env import EmbeddingEnv
        from workers.retrieval_env import RetrievalEnv
        from workers.evaluation_env import EvaluationEnv
        
        worker_instances = {
            "worker_1": DataQualityTriageEnv(task_id=task_map["worker_1"]),
            "worker_2": ChunkingEnv(),
            "worker_3": EmbeddingEnv(),
            "worker_4": RetrievalEnv(),
            "worker_5": EvaluationEnv(),
        }
        
        for wid, worker in worker_instances.items():
            self.registry.register(wid, worker)
        
        self.registry.reset_all(task_map)
        
        anomaly_config = ANOMALY_INJECTION_MAP.get(self.task_id, [])
        self.anomaly_injector.inject(worker_instances, anomaly_config)
        
        for worker_id, anomaly_type in anomaly_config:
            from fleet.models import AnomalyType
            if anomaly_type != AnomalyType.NONE:
                self.registry.set_anomaly_flag(worker_id, 1)
        
        self.episode_phase = EpisodePhase.OVERSIGHT

    def _check_missed_violations(self) -> None:
        """At episode end, penalize any undetected anomalies."""
        actual_anomalies = set(self.anomaly_injector.get_all_anomalies().keys())
        detected = {
            e.worker_id for e in self.governance.audit_trail
            if e.action_type in ("intervene", "escalate") and e.was_correct
        }
        for wid in actual_anomalies - detected:
            self.governance.record_missed_violation(wid, self.step_count)
            self.total_reward += -0.65   # Missed violation penalty applied at episode end

    def _compute_final_pipeline_quality(self) -> None:
        """Compute pipeline quality from Worker 5 final score."""
        w5 = self.registry.workers.get("worker_5")
        if w5 and hasattr(w5, "_compute_final_score"):
            self.pipeline_quality_score = w5._compute_final_score()
        else:
            self.pipeline_quality_score = 0.5   # Default if not available

    def _compute_explainability(self) -> float:
        if not self.audit_report:
            return 0.0
        return compute_explainability_score(self.audit_report, self.anomaly_injector)

    def close(self) -> None:
        """Cleanup."""
        pass
