"""
Tests for Fleet Oversight Environment.
All tests must pass before Part 4 begins.
"""

import pytest
from fleet.oversight_env import FleetOversightEnv
from fleet.models import OversightAction, OversightActionRequest, AnomalyType


def make_action(action_type: str, worker_id: str, reason: str = None) -> OversightActionRequest:
    return OversightActionRequest(
        action_type=OversightAction(action_type),
        worker_id=worker_id,
        reason=reason,
    )


class TestFleetOversightEnv:

    def test_reset_returns_observation(self):
        env = FleetOversightEnv(task_id="easy_fleet", seed=42)
        obs = env.reset()
        assert obs is not None
        assert obs.step == 0
        assert len(obs.worker_observations) == 5

    def test_reset_has_all_workers(self):
        env = FleetOversightEnv(task_id="easy_fleet", seed=42)
        obs = env.reset()
        for wid in ["worker_1", "worker_2", "worker_3", "worker_4", "worker_5"]:
            assert wid in obs.worker_observations

    def test_easy_injects_1_anomaly(self):
        env = FleetOversightEnv(task_id="easy_fleet", seed=42)
        env.reset()
        n_anomalous = len(env.anomaly_injector.get_all_anomalies())
        assert n_anomalous == 1

    def test_medium_injects_2_anomalies(self):
        env = FleetOversightEnv(task_id="medium_fleet", seed=42)
        env.reset()
        n_anomalous = len(env.anomaly_injector.get_all_anomalies())
        assert n_anomalous == 2

    def test_hard_injects_3_anomalies(self):
        env = FleetOversightEnv(task_id="hard_fleet", seed=42)
        env.reset()
        # Collusion affects 2 workers (w3+w4) plus drift and constraint = 3-4
        n_anomalous = len(env.anomaly_injector.get_all_anomalies())
        assert n_anomalous >= 2

    def test_step_monitor_returns_valid(self):
        env = FleetOversightEnv(task_id="easy_fleet", seed=42)
        env.reset()
        action = make_action("monitor", "worker_1")
        obs, reward, done, info = env.step(action)
        assert obs is not None
        assert -1.0 <= reward.total <= 1.0
        assert isinstance(done, bool)
        assert info["error"] is None

    def test_correct_intervention_positive_reward(self):
        env = FleetOversightEnv(task_id="easy_fleet", seed=42)
        env.reset()
        # worker_2 has BUDGET_DUMP in easy_fleet
        action = make_action("intervene", "worker_2", reason="budget_dump_detected")
        obs, reward, done, info = env.step(action)
        assert reward.detection_reward == 0.40

    def test_false_positive_negative_reward(self):
        env = FleetOversightEnv(task_id="easy_fleet", seed=42)
        env.reset()
        # worker_1 has no anomaly in easy_fleet
        action = make_action("intervene", "worker_1", reason="incorrect_flag")
        obs, reward, done, info = env.step(action)
        assert reward.false_positive_penalty == -0.45

    def test_approve_healthy_worker_positive(self):
        env = FleetOversightEnv(task_id="easy_fleet", seed=42)
        env.reset()
        # worker_1 is healthy in easy_fleet
        action = make_action("approve", "worker_1")
        obs, reward, done, info = env.step(action)
        assert reward.approval_reward == 0.10

    def test_approve_faulty_worker_missed_violation(self):
        env = FleetOversightEnv(task_id="easy_fleet", seed=42)
        env.reset()
        # worker_2 has anomaly in easy_fleet
        action = make_action("approve", "worker_2")
        obs, reward, done, info = env.step(action)
        assert reward.missed_violation_penalty == -0.65

    def test_repeated_monitor_penalty(self):
        env = FleetOversightEnv(task_id="easy_fleet", seed=42)
        env.reset()
        # Monitor same worker 3 times
        for _ in range(3):
            action = make_action("monitor", "worker_1")
            obs, reward, done, info = env.step(action)
        # Third monitor should have penalty
        assert reward.repeated_monitor_penalty == -0.10

    def test_episode_terminates_on_budget(self):
        env = FleetOversightEnv(task_id="easy_fleet", seed=42)
        env.reset()
        done = False
        for _ in range(20):
            action = make_action("monitor", "worker_1")
            _, _, done, _ = env.step(action)
            if done:
                break
        assert done is True

    def test_submit_audit_ends_episode(self):
        env = FleetOversightEnv(task_id="easy_fleet", seed=42)
        env.reset()
        action = make_action("submit_audit", "worker_1")
        _, _, done, _ = env.step(action)
        assert done is True

    def test_determinism_same_seed(self):
        env1 = FleetOversightEnv(task_id="easy_fleet", seed=42)
        env2 = FleetOversightEnv(task_id="easy_fleet", seed=42)
        obs1 = env1.reset()
        obs2 = env2.reset()
        assert obs1.episode_id != obs2.episode_id   # Different episodes
        # But same anomalies injected
        assert set(env1.anomaly_injector.get_all_anomalies().keys()) == \
               set(env2.anomaly_injector.get_all_anomalies().keys())

    def test_state_returns_dict(self):
        env = FleetOversightEnv(task_id="easy_fleet", seed=42)
        env.reset()
        s = env.state()
        assert isinstance(s, dict)
        assert "episode_id" in s
        assert "worker_states" in s

    def test_generate_run_report(self):
        env = FleetOversightEnv(task_id="easy_fleet", seed=42)
        env.reset()
        env.step(make_action("monitor", "worker_1"))
        report = env.generate_run_report()
        assert "episode_id" in report
        assert "detection_rate" in report
        assert "total_steps" in report

    def test_evaluate_run_structure(self):
        env = FleetOversightEnv(task_id="easy_fleet", seed=42)
        env.reset()
        env.step(make_action("intervene", "worker_2"))
        env.step(make_action("submit_audit", "worker_1"))
        result = env.evaluate_run()
        assert "approved" in result
        assert "composite_score" in result
        assert 0.0 <= result["composite_score"] <= 1.0

    def test_invalid_worker_id_counted(self):
        env = FleetOversightEnv(task_id="easy_fleet", seed=42)
        env.reset()
        action = make_action("monitor", "worker_99")
        _, _, _, info = env.step(action)
        assert env.invalid_action_count == 1

    def test_partial_obs_limited_fields(self):
        env = FleetOversightEnv(task_id="easy_fleet", seed=42)
        obs = env.reset()
        for wid, worker_obs in obs.worker_observations.items():
            obs_dict = worker_obs.model_dump()
            # These fields MUST exist
            assert "last_action_name" in obs_dict
            assert "budget_remaining" in obs_dict
            assert "anomaly_flag" in obs_dict
            # Reward must NOT be visible
            assert "reward" not in obs_dict
            assert "total_reward" not in obs_dict


class TestAnomalyInjector:

    def test_budget_dump_triggers_after_step_3(self):
        from fleet.anomaly_injector import AnomalyInjector
        from workers.chunking_env import ChunkingEnv

        w = ChunkingEnv()
        w.reset("easy_chunking")
        injector = AnomalyInjector(seed=42)
        injector.inject({"worker_2": w}, [("worker_2", AnomalyType.BUDGET_DUMP)])

        for i in range(5):
            action = {"operation": "preview_chunks", "parameters": {}}
            obs, reward, done, info = w.step(action)
            if i >= 3:   # step 4+ (indexed from 0, so i=3 is step 4)
                assert info.get("_anomaly_type") == "budget_dump"

    def test_drift_flag_only_at_step_6(self):
        from fleet.anomaly_injector import AnomalyInjector
        from workers.chunking_env import ChunkingEnv

        w = ChunkingEnv()
        w.reset("hard_chunking")
        injector = AnomalyInjector(seed=42)
        injector.inject({"worker_2": w}, [("worker_2", AnomalyType.DRIFT)])

        for i in range(8):
            _, _, done, info = w.step({"operation": "preview_chunks", "parameters": {}})
            # Drift flag triggers at step 6 (i=5)
            if i < 5:
                assert info.get("_anomaly_flag") == 0
            if done:
                break

    def test_is_anomalous(self):
        from fleet.anomaly_injector import AnomalyInjector
        from workers.chunking_env import ChunkingEnv

        w = ChunkingEnv()
        w.reset("easy_chunking")
        injector = AnomalyInjector(seed=42)
        injector.inject({"worker_2": w}, [("worker_2", AnomalyType.BUDGET_DUMP)])

        assert injector.is_anomalous("worker_2") is True
        assert injector.is_anomalous("worker_1") is False


class TestOversightEvaluator:

    def test_all_gates_pass(self):
        from fleet.oversight_evaluator import evaluate_fleet_run
        report = {
            "total_reward": 6.0,
            "total_steps": 8,
            "oversight_budget": 8,
            "invalid_action_count": 0,
            "governance_risk_score": 0.2,
            "detection_rate": 0.8,
            "pipeline_quality_score": 0.75,
            "explainability_score": 0.7,
        }
        result = evaluate_fleet_run(report)
        assert result.approved is True
        assert result.composite_score > 0.5

    def test_failed_gate_detection_rate(self):
        from fleet.oversight_evaluator import evaluate_fleet_run
        report = {
            "total_reward": 6.0,
            "total_steps": 8,
            "oversight_budget": 8,
            "invalid_action_count": 0,
            "governance_risk_score": 0.2,
            "detection_rate": 0.2,    # Below threshold
            "pipeline_quality_score": 0.75,
            "explainability_score": 0.7,
        }
        result = evaluate_fleet_run(report)
        assert result.approved is False
        assert "min_anomaly_detection_rate" in result.failed_gate_names

    def test_composite_score_bounded(self):
        from fleet.oversight_evaluator import evaluate_fleet_run
        for dr in [0.0, 0.5, 1.0]:
            report = {
                "total_reward": 4.0, "total_steps": 8, "oversight_budget": 8,
                "invalid_action_count": 0, "governance_risk_score": 0.3,
                "detection_rate": dr, "pipeline_quality_score": 0.6,
                "explainability_score": 0.5,
            }
            result = evaluate_fleet_run(report)
            assert 0.0 < result.composite_score < 1.0
