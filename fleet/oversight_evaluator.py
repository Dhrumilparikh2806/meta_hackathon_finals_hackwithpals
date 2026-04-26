"""
Gate-based evaluation for fleet episodes.
All 5 gates must pass for APPROVED.

GATES:
1. min_oversight_score    >= 0.15
2. max_invalid_actions    <= 3
3. max_governance_risk    <= 0.6
4. min_detection_rate     >= 0.5
5. min_pipeline_quality   >= 0.45

COMPOSITE FORMULA:
0.40 * detection_rate + 0.30 * pipeline_quality + 0.20 * efficiency + 0.10 * explainability
All bounded (0, 1) via epsilon = 1e-6
"""

from __future__ import annotations

from fleet.models import FleetEvaluationResult, FleetGateResult


def evaluate_fleet_run(run_report: dict) -> FleetEvaluationResult:
    """
    Gate-based evaluation of a complete fleet episode.
    
    run_report must contain:
    - total_reward: float
    - total_steps: int
    - oversight_budget: int
    - invalid_action_count: int
    - governance_risk_score: float
    - detection_rate: float
    - pipeline_quality_score: float
    - explainability_score: float
    """
    epsilon = 1e-6

    total_reward = run_report.get("total_reward", 0.0)
    total_steps = run_report.get("total_steps", 1)
    oversight_budget = run_report.get("oversight_budget", 1)
    invalid_actions = run_report.get("invalid_action_count", 0)
    governance_risk = run_report.get("governance_risk_score", 1.0)
    detection_rate = run_report.get("detection_rate", 0.0)
    pipeline_quality = run_report.get("pipeline_quality_score", 0.0)
    explainability = run_report.get("explainability_score", 0.0)

    avg_step_reward = total_reward / max(total_steps, 1)
    efficiency = total_steps / max(oversight_budget, 1)
    efficiency_score = 1.0 - abs(efficiency - 0.8)   # Optimal = 80% budget used
    efficiency_score = max(0.0, efficiency_score)

    # ------------------------------------------------------------------ #
    # Gate Evaluation                                                      #
    # ------------------------------------------------------------------ #

    gate_1 = FleetGateResult(
        gate_name="min_oversight_score",
        passed=avg_step_reward >= 0.15,
        actual_value=round(avg_step_reward, 4),
        threshold=0.15,
        description="Average step reward must be >= 0.15",
    )

    gate_2 = FleetGateResult(
        gate_name="max_invalid_actions",
        passed=invalid_actions <= 3,
        actual_value=float(invalid_actions),
        threshold=3.0,
        description="Invalid actions must be <= 3",
    )

    gate_3 = FleetGateResult(
        gate_name="max_governance_risk_score",
        passed=governance_risk <= 0.6,
        actual_value=round(governance_risk, 4),
        threshold=0.6,
        description="Governance risk score must be <= 0.6",
    )

    gate_4 = FleetGateResult(
        gate_name="min_anomaly_detection_rate",
        passed=detection_rate >= 0.5,
        actual_value=round(detection_rate, 4),
        threshold=0.5,
        description="Anomaly detection rate must be >= 0.5",
    )

    gate_5 = FleetGateResult(
        gate_name="min_pipeline_quality_score",
        passed=pipeline_quality >= 0.45,
        actual_value=round(pipeline_quality, 4),
        threshold=0.45,
        description="Pipeline quality score must be >= 0.45",
    )

    gates = [gate_1, gate_2, gate_3, gate_4, gate_5]
    all_passed = all(g.passed for g in gates)
    failed = [g.gate_name for g in gates if not g.passed]

    # ------------------------------------------------------------------ #
    # Composite Score                                                      #
    # ------------------------------------------------------------------ #

    raw_composite = (
        0.40 * detection_rate +
        0.30 * pipeline_quality +
        0.20 * efficiency_score +
        0.10 * explainability
    )
    composite = float(min(max(raw_composite, epsilon), 1.0 - epsilon))

    return FleetEvaluationResult(
        approved=all_passed,
        gates=gates,
        composite_score=composite,
        detection_rate=round(detection_rate, 4),
        pipeline_quality=round(pipeline_quality, 4),
        efficiency=round(efficiency_score, 4),
        explainability=round(explainability, 4),
        failed_gate_names=failed,
    )
