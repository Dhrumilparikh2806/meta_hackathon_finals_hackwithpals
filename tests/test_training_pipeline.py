"""
Tests for training pipeline components.
Does not require GPU — tests environment integration only.
"""

import pytest
import json
from pathlib import Path
from fleet.oversight_env import FleetOversightEnv
from fleet.models import OversightAction, OversightActionRequest
from fleet_train import (
    parse_action_from_text,
    build_prompt,
    run_random_baseline,
    VALID_ACTIONS,
    VALID_WORKERS,
)


class TestParseAction:

    def test_valid_json(self):
        text = '{"action_type": "intervene", "worker_id": "worker_2", "reason": "test"}'
        action, worker, reason = parse_action_from_text(text)
        assert action == "intervene"
        assert worker == "worker_2"
        assert reason == "test"

    def test_markdown_fenced_json(self):
        text = '```json\n{"action_type": "monitor", "worker_id": "worker_1"}\n```'
        action, worker, reason = parse_action_from_text(text)
        assert action == "monitor"
        assert worker == "worker_1"

    def test_invalid_json_fallback(self):
        text = "I think we should monitor worker 1"
        action, worker, reason = parse_action_from_text(text)
        assert action == "monitor"
        assert worker == "worker_1"

    def test_partial_json(self):
        text = 'The action is: {"action_type": "approve", "worker_id": "worker_3"}'
        action, worker, reason = parse_action_from_text(text)
        assert action == "approve"
        assert worker == "worker_3"


class TestBuildPrompt:

    def test_prompt_contains_step(self):
        env = FleetOversightEnv(task_id="easy_fleet", seed=42)
        obs = env.reset()
        obs_dict = obs.model_dump()
        prompt = build_prompt(1, obs_dict, 0.0, [])
        assert "Step: 1" in prompt

    def test_prompt_contains_budget(self):
        env = FleetOversightEnv(task_id="easy_fleet", seed=42)
        obs = env.reset()
        obs_dict = obs.model_dump()
        prompt = build_prompt(1, obs_dict, 0.0, [])
        assert "Budget:" in prompt

    def test_prompt_with_history(self):
        env = FleetOversightEnv(task_id="easy_fleet", seed=42)
        obs = env.reset()
        obs_dict = obs.model_dump()
        history = ["Step 1: monitor→worker_1 = +0.000"]
        prompt = build_prompt(2, obs_dict, 0.1, history)
        assert "monitor" in prompt


class TestRandomBaseline:

    def test_baseline_returns_dict(self):
        env = FleetOversightEnv(task_id="easy_fleet", seed=42)
        result = run_random_baseline(env, n_episodes=2)
        assert isinstance(result, dict)
        assert "mean_detection_rate" in result
        assert "mean_total_reward" in result

    def test_baseline_detection_in_range(self):
        env = FleetOversightEnv(task_id="easy_fleet", seed=42)
        result = run_random_baseline(env, n_episodes=3)
        assert 0.0 <= result["mean_detection_rate"] <= 1.0

    def test_baseline_n_results(self):
        env = FleetOversightEnv(task_id="easy_fleet", seed=42)
        result = run_random_baseline(env, n_episodes=3)
        assert len(result["detection_rates"]) == 3
        assert len(result["total_rewards"]) == 3


class TestSimulationMode:

    def test_simulation_generates_plots(self, tmp_path, monkeypatch):
        import fleet_train
        monkeypatch.setattr(fleet_train, "PLOTS_DIR", tmp_path)
        fleet_train._run_simulation_training("easy_fleet", 5)
        assert (tmp_path / "reward_curve.png").exists()
        assert (tmp_path / "detection_rate.png").exists()
        assert (tmp_path / "before_after.png").exists()
        assert (tmp_path / "training_metrics.json").exists()

    def test_simulation_metrics_json(self, tmp_path, monkeypatch):
        import fleet_train
        monkeypatch.setattr(fleet_train, "PLOTS_DIR", tmp_path)
        fleet_train._run_simulation_training("easy_fleet", 5)
        with open(tmp_path / "training_metrics.json") as f:
            metrics = json.load(f)
        assert "episode_rewards" in metrics
        assert "detection_rates" in metrics
        assert len(metrics["episode_rewards"]) == 5
        assert 0 <= metrics["final_detection_rate"] <= 1.0


class TestRewardFunctionIntegration:

    def test_intervene_correct_worker_positive(self):
        env = FleetOversightEnv(task_id="easy_fleet", seed=42)
        env.reset()
        # worker_2 has BUDGET_DUMP in easy_fleet
        action = OversightActionRequest(
            action_type=OversightAction.INTERVENE,
            worker_id="worker_2",
            reason="budget dump",
        )
        _, reward_obj, _, _ = env.step(action)
        assert reward_obj.total > 0

    def test_intervene_wrong_worker_negative(self):
        env = FleetOversightEnv(task_id="easy_fleet", seed=42)
        env.reset()
        # worker_1 is healthy in easy_fleet
        action = OversightActionRequest(
            action_type=OversightAction.INTERVENE,
            worker_id="worker_1",
        )
        _, reward_obj, _, _ = env.step(action)
        assert reward_obj.total < 0
