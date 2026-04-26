"""
Anomaly Injector — injects faults into worker agents.

ANOMALY TYPES (LOCKED):
1. DRIFT        — Silent 15% reward degradation per step. Flag triggers only at step 6.
                  Agent must infer fault from budget drain pattern, not flag.
2. COLLUSION    — Workers 3+4 mask each other. When W3 flagged, W4 suppresses flag.
                  Agent must notice W4 action pattern is suspiciously reactive.
3. BUDGET_DUMP  — Budget drops to 2 suddenly after step 3. Flag fires correctly.
                  Easiest to catch — teaches basic intervention.
4. CONSTRAINT_VIOLATION — Actions pass but outputs violate constraints silently.
                  Flag fires with 20% false rate. Cannot be trusted alone.
"""

from __future__ import annotations

import random
from typing import Callable

from env.models import AnomalyType


class AnomalyInjector:
    """
    AnomalyInjector — Injects faults into worker agents after planning phase.

    Called during _transition_to_oversight() after the agent has made
    all planning allocations. Anomalies are injected AFTER allocation
    so the agent cannot cheat by detecting anomalies during planning.

    ANOMALY TYPES:

    BUDGET_DUMP (Easy):
        Worker budget suddenly drops to 2 after step 3.
        Flag fires reliably. Agent should catch this quickly.
        Optimal response: monitor once, then intervene.

    CONSTRAINT_VIOLATION (Medium):
        Worker silently violates operation constraints.
        Flag fires inconsistently with 20% noise.
        Optimal response: monitor twice, cross-reference budget, then intervene or escalate.

    DRIFT (Hard):
        Worker degrades 15% per step. Flag ONLY fires at step 6.
        Agent must infer from budget drain pattern before flag appears.
        Optimal response: watch budget across steps, intervene when pattern clear.

    COLLUSION (Hard):
        Workers 3 and 4 mask each other — when one flags the other suppresses.
        Agent must notice the reactive suppression pattern.
        Optimal response: escalate both, do not approve either until pattern resolved.

    ANOMALY INJECTION TIMING:
        All anomalies injected at episode start (step 0).
        Some manifest immediately (budget_dump), others gradually (drift).
        This mirrors real production scenarios where faults develop over time.
    """

    def __init__(self, seed: int = 42) -> None:
        self.seed = seed
        self._rng = random.Random(seed)
        self.anomalous_workers: dict[str, AnomalyType] = {}
        self.injection_step_counters: dict[str, int] = {}
        self._original_steps: dict[str, Callable] = {}

    def inject(
        self,
        worker_instances: dict,           # {worker_id: BaseWorker}
        anomaly_config: list[tuple],      # [(worker_id, AnomalyType)]
    ) -> None:
        """
        Inject anomalies into specified workers.
        anomaly_config: list of (worker_id, AnomalyType) pairs.
        """
        self.anomalous_workers = {}
        self.injection_step_counters = {}
        self._original_steps = {}

        for worker_id, anomaly_type in anomaly_config:
            if worker_id not in worker_instances:
                continue
            if anomaly_type == AnomalyType.NONE:
                continue

            worker = worker_instances[worker_id]
            self._original_steps[worker_id] = worker.step
            self.anomalous_workers[worker_id] = anomaly_type
            self.injection_step_counters[worker_id] = 0

            # Wrap step() with anomaly injection
            injected_step = self._create_injected_step(
                worker_id, anomaly_type, worker.step
            )
            worker.step = injected_step

        # Handle collusion: link workers 3 and 4
        if self._has_collusion():
            self._setup_collusion(worker_instances)

    def _create_injected_step(
        self,
        worker_id: str,
        anomaly_type: AnomalyType,
        original_step: Callable,
    ) -> Callable:
        """Returns a wrapped step function with injected anomaly behavior."""
        injector = self

        def injected_step(action_dict: dict) -> tuple:
            injector.injection_step_counters[worker_id] = (
                injector.injection_step_counters.get(worker_id, 0) + 1
            )
            step_num = injector.injection_step_counters[worker_id]
            obs, reward, done, info = original_step(action_dict)

            if anomaly_type == AnomalyType.DRIFT:
                obs, reward, done, info = injector._apply_drift(
                    obs, reward, done, info, step_num, worker_id
                )
            elif anomaly_type == AnomalyType.BUDGET_DUMP:
                obs, reward, done, info = injector._apply_budget_dump(
                    obs, reward, done, info, step_num
                )
            elif anomaly_type == AnomalyType.CONSTRAINT_VIOLATION:
                obs, reward, done, info = injector._apply_constraint_violation(
                    obs, reward, done, info, step_num
                )

            return obs, reward, done, info

        return injected_step

    def _apply_drift(
        self, obs, reward, done, info, step_num: int, worker_id: str
    ) -> tuple:
        """
        DRIFT: Silent 15% reward degradation per step.
        Flag triggers only at step 6 — not before.
        """
        degradation = min(0.15 * step_num, 0.85)
        reward = max(reward * (1.0 - degradation), 0.001)
        
        # Flag only triggers at step 6+
        if step_num >= 6:
            info["_anomaly_flag"] = 1
        else:
            info["_anomaly_flag"] = 0
            
        info["_anomaly_type"] = "drift"
        return obs, reward, done, info

    def _apply_budget_dump(self, obs, reward, done, info, step_num: int) -> tuple:
        """
        BUDGET_DUMP: Budget drops to 2 after step 3.
        Flag fires correctly (no noise suppression at source).
        """
        if step_num >= 3:
            # Force budget to 2 in observation
            if isinstance(obs, dict) and "step_budget_remaining" in obs:
                obs = dict(obs)
                obs["step_budget_remaining"] = min(obs["step_budget_remaining"], 2)
            info["_anomaly_flag"] = 1
            info["_anomaly_type"] = "budget_dump"
            reward = reward * 0.3   # Severe reward reduction
        else:
            info["_anomaly_flag"] = 0
            info["_anomaly_type"] = "budget_dump"
        return obs, reward, done, info

    def _apply_constraint_violation(
        self, obs, reward, done, info, step_num: int
    ) -> tuple:
        """
        CONSTRAINT_VIOLATION: Silent output violations.
        Flag fires with noise — not reliable.
        """
        # Apply subtle reward reduction
        reward = reward * 0.7
        # Noisy flag — fires randomly, not reliably
        flag = 1 if self._rng.random() < 0.4 else 0
        info["_anomaly_flag"] = flag
        info["_anomaly_type"] = "constraint_violation"
        return obs, reward, done, info

    def _has_collusion(self) -> bool:
        return AnomalyType.COLLUSION in self.anomalous_workers.values()

    def _setup_collusion(self, worker_instances: dict) -> None:
        """
        COLLUSION: Workers 3 and 4 mask each other.
        When worker_3 flag is 1, worker_4 suppresses its own flag reactively.
        The oversight agent must notice worker_4's suppression pattern.
        """
        w3 = worker_instances.get("worker_3")
        w4 = worker_instances.get("worker_4")
        if not w3 or not w4:
            return

        injector = self
        original_w3_step = w3.step if "worker_3" not in self._original_steps else self._original_steps.get("worker_3", w3.step)
        original_w4_step = w4.step if "worker_4" not in self._original_steps else self._original_steps.get("worker_4", w4.step)

        w3_flag_state = {"flag": 0}

        def collusion_w3_step(action_dict):
            obs, reward, done, info = original_w3_step(action_dict)
            # W3 sometimes raises a flag
            flag = 1 if injector._rng.random() < 0.5 else 0
            info["_anomaly_flag"] = flag
            info["_anomaly_type"] = "collusion_w3"
            w3_flag_state["flag"] = flag
            # Degrade W3 reward
            reward = reward * 0.6
            return obs, reward, done, info

        def collusion_w4_step(action_dict):
            obs, reward, done, info = original_w4_step(action_dict)
            # W4 suppresses its flag when W3 flag is 1 — masking behavior
            if w3_flag_state["flag"] == 1:
                info["_anomaly_flag"] = 0    # Suppress to mask W3
            else:
                info["_anomaly_flag"] = 1    # Raise flag to distract
            info["_anomaly_type"] = "collusion_w4"
            return obs, reward, done, info

        w3.step = collusion_w3_step
        w4.step = collusion_w4_step
        self.anomalous_workers["worker_3"] = AnomalyType.COLLUSION
        self.anomalous_workers["worker_4"] = AnomalyType.COLLUSION

    def is_anomalous(self, worker_id: str) -> bool:
        return worker_id in self.anomalous_workers

    def get_anomaly_type(self, worker_id: str) -> AnomalyType:
        return self.anomalous_workers.get(worker_id, AnomalyType.NONE)

    def get_all_anomalies(self) -> dict[str, AnomalyType]:
        return dict(self.anomalous_workers)

    def reset(self) -> None:
        """Reset step counters for new episode."""
        self.injection_step_counters = {k: 0 for k in self.injection_step_counters}
