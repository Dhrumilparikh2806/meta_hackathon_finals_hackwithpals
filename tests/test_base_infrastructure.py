"""
Tests for base infrastructure — Part 1.
All tests must pass before Part 2 begins.
"""

import pytest
from unittest.mock import MagicMock
from workers.base_worker import BaseWorker
from env.models import (
    WorkerStatus,
    AnomalyType,
    OversightAction,
    Difficulty,
    WorkerPartialObservation,
    FleetObservation,
    OversightActionRequest,
    OversightReward,
    FleetTaskConfig,
    FLEET_TASK_CONFIGS,
    FleetEvaluationResult,
    FleetGateResult,
    AuditEvent,
    FleetAuditReport,
)


# ------------------------------------------------------------------ #
# Concrete implementation for testing BaseWorker                       #
# ------------------------------------------------------------------ #

class ConcreteWorker(BaseWorker):
    """Minimal concrete implementation for testing BaseWorker."""

    def reset(self, task_id: str) -> dict:
        self._reset_episode_tracking()
        self.task_id = task_id
        self.step_budget = 8
        self.step_budget_remaining = 8
        return self.state()

    def step(self, action_dict: dict) -> tuple[dict, float, bool, dict]:
        self.step_count += 1
        self.step_budget_remaining -= 1
        reward = self._clip_reward(0.5)
        self.total_reward += reward
        done = self._is_budget_exhausted()
        info = {"error": None, "action": action_dict.get("operation", "unknown")}
        self._record_action(info["action"], reward, info)
        return self.state(), reward, done, info

    def state(self) -> dict:
        return self._get_base_state()

    def generate_run_report(self) -> dict:
        return self._get_base_report()

    def evaluate_run(self) -> dict:
        epsilon = 1e-6
        score = min(max(self.total_reward / max(self.step_count, 1), epsilon), 1.0 - epsilon)
        return {
            "approved": score >= 0.5,
            "gates": {},
            "composite_score": score,
        }


# ------------------------------------------------------------------ #
# BaseWorker Tests                                                     #
# ------------------------------------------------------------------ #

class TestBaseWorker:

    def test_instantiation(self):
        w = ConcreteWorker("worker_1", "Test Worker")
        assert w.worker_id == "worker_1"
        assert w.worker_name == "Test Worker"

    def test_reset_returns_dict(self):
        w = ConcreteWorker("worker_1", "Test Worker")
        obs = w.reset("easy_task")
        assert isinstance(obs, dict)
        assert obs["task_id"] == "easy_task"
        assert obs["step_count"] == 0

    def test_step_returns_tuple(self):
        w = ConcreteWorker("worker_1", "Test Worker")
        w.reset("easy_task")
        result = w.step({"operation": "test"})
        assert isinstance(result, tuple)
        assert len(result) == 4

    def test_reward_always_in_range(self):
        w = ConcreteWorker("worker_1", "Test Worker")
        w.reset("easy_task")
        for _ in range(5):
            _, reward, _, _ = w.step({"operation": "test"})
            assert 0.0 <= reward <= 1.0, f"Reward out of range: {reward}"

    def test_budget_exhaustion_terminates(self):
        w = ConcreteWorker("worker_1", "Test Worker")
        w.reset("easy_task")
        done = False
        for _ in range(10):
            _, _, done, _ = w.step({"operation": "test"})
            if done:
                break
        assert done is True

    def test_clip_reward(self):
        w = ConcreteWorker("worker_1", "Test Worker")
        assert w._clip_reward(0.5) == pytest.approx(0.5, abs=1e-4)
        assert w._clip_reward(-1.0) > 0.0
        assert w._clip_reward(2.0) < 1.0

    def test_governance_event_recording(self):
        w = ConcreteWorker("worker_1", "Test Worker")
        w.reset("easy_task")
        w._record_governance_event("test_event", "high", "test detail")
        assert len(w.governance_events) == 1
        assert w.governance_events[0]["event_type"] == "test_event"

    def test_invalid_governance_severity_raises(self):
        w = ConcreteWorker("worker_1", "Test Worker")
        w.reset("easy_task")
        with pytest.raises(AssertionError):
            w._record_governance_event("test", "invalid_severity", "detail")

    def test_state_has_required_keys(self):
        w = ConcreteWorker("worker_1", "Test Worker")
        w.reset("easy_task")
        s = w.state()
        required_keys = [
            "worker_id", "worker_name", "task_id", "step_count",
            "step_budget_remaining", "total_reward", "is_done", "submitted"
        ]
        for key in required_keys:
            assert key in s, f"Missing key: {key}"

    def test_run_report_has_required_keys(self):
        w = ConcreteWorker("worker_1", "Test Worker")
        w.reset("easy_task")
        w.step({"operation": "test"})
        report = w.generate_run_report()
        required_keys = [
            "worker_id", "task_id", "step_count", "total_reward",
            "action_history", "governance_events", "submitted"
        ]
        for key in required_keys:
            assert key in report, f"Missing key: {key}"

    def test_determinism_same_seed(self):
        w1 = ConcreteWorker("worker_1", "Test Worker")
        w2 = ConcreteWorker("worker_1", "Test Worker")
        obs1 = w1.reset("easy_task")
        obs2 = w2.reset("easy_task")
        assert obs1 == obs2

    def test_reset_clears_history(self):
        w = ConcreteWorker("worker_1", "Test Worker")
        w.reset("easy_task")
        w.step({"operation": "test"})
        w.reset("easy_task")
        assert w.step_count == 0
        assert w.total_reward == 0.0
        assert len(w.action_history) == 0


# ------------------------------------------------------------------ #
# Fleet Models Tests                                                   #
# ------------------------------------------------------------------ #

class TestFleetModels:

    def test_worker_partial_observation(self):
        obs = WorkerPartialObservation(
            worker_id="worker_1",
            worker_name="Data Clean",
            status=WorkerStatus.RUNNING,
            last_action_name="inspect_schema",
            budget_remaining=8,
            anomaly_flag=0,
            step_count=1,
        )
        assert obs.worker_id == "worker_1"
        assert obs.anomaly_flag == 0

    def test_anomaly_flag_bounds(self):
        with pytest.raises(Exception):
            WorkerPartialObservation(
                worker_id="w1", worker_name="W1",
                status=WorkerStatus.RUNNING,
                budget_remaining=8, anomaly_flag=2, step_count=0
            )

    def test_oversight_reward_total_clipped(self):
        r = OversightReward(
            detection_reward=0.40,
            false_positive_penalty=-0.45,
            missed_violation_penalty=-0.65,
        )
        assert -1.0 <= r.total <= 1.0

    def test_oversight_reward_positive_case(self):
        r = OversightReward(detection_reward=0.40, completion_bonus=0.20)
        assert r.total > 0

    def test_fleet_task_configs_exist(self):
        assert "easy_fleet" in FLEET_TASK_CONFIGS
        assert "medium_fleet" in FLEET_TASK_CONFIGS
        assert "hard_fleet" in FLEET_TASK_CONFIGS

    def test_easy_config_values(self):
        cfg = FLEET_TASK_CONFIGS["easy_fleet"]
        assert cfg.oversight_budget == 8
        assert cfg.n_anomalous_workers == 1
        assert AnomalyType.BUDGET_DUMP in cfg.anomaly_types

    def test_medium_config_values(self):
        cfg = FLEET_TASK_CONFIGS["medium_fleet"]
        assert cfg.oversight_budget == 12
        assert cfg.n_anomalous_workers == 2
        assert AnomalyType.BUDGET_DUMP in cfg.anomaly_types
        assert AnomalyType.CONSTRAINT_VIOLATION in cfg.anomaly_types

    def test_hard_config_values(self):
        cfg = FLEET_TASK_CONFIGS["hard_fleet"]
        assert cfg.oversight_budget == 16
        assert cfg.n_anomalous_workers == 3
        assert AnomalyType.DRIFT in cfg.anomaly_types
        assert AnomalyType.COLLUSION in cfg.anomaly_types

    def test_oversight_action_enum_values(self):
        assert OversightAction.MONITOR == "monitor"
        assert OversightAction.INTERVENE == "intervene"
        assert OversightAction.SUBMIT_AUDIT == "submit_audit"

    def test_fleet_observation_model(self):
        worker_obs = {
            "worker_1": WorkerPartialObservation(
                worker_id="worker_1", worker_name="Data Clean",
                status=WorkerStatus.RUNNING, budget_remaining=8,
                anomaly_flag=0, step_count=0
            )
        }
        fleet_obs = FleetObservation(
            step=1,
            oversight_budget_remaining=8,
            worker_observations=worker_obs,
            anomaly_alerts=[],
            episode_id="test_episode_001"
        )
        assert fleet_obs.step == 1
        assert "worker_1" in fleet_obs.worker_observations


# ------------------------------------------------------------------ #
# requirements.txt additions — verify imports work                    #
# ------------------------------------------------------------------ #

class TestImports:

    def test_pydantic_import(self):
        from pydantic import BaseModel
        assert BaseModel is not None

    def test_workers_package_import(self):
        import workers
        assert workers is not None

    def test_fleet_package_import(self):
        import fleet
        assert fleet is not None

    def test_base_worker_import(self):
        from workers.base_worker import BaseWorker
        assert BaseWorker is not None

    def test_fleet_models_import(self):
        from env.models import FleetObservation, OversightReward
        assert FleetObservation is not None
        assert OversightReward is not None
