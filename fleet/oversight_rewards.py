"""
Oversight Agent Reward Computation.

REWARD TABLE (LOCKED — DO NOT CHANGE VALUES):
+0.40  true_detection     — caught fault, intervened correctly
+0.10  correct_approval   — approved healthy worker correctly
+0.15  correct_escalation — escalated ambiguous worker
+0.08  explainability     — audit report quality score
+0.20  completion_bonus   — episode completion
-0.45  false_positive     — intervened on healthy worker
-0.65  missed_violation   — fault propagated unchecked
-0.10  repeated_monitor   — unnecessary repeat monitor on same worker
"""

from __future__ import annotations

from fleet.models import OversightAction, OversightReward


def compute_oversight_step_reward(
    action_type: OversightAction,
    worker_id: str,
    anomaly_injector,               # AnomalyInjector instance
    worker_registry,                # WorkerRegistry instance
    monitor_history: dict[str, int],  # {worker_id: times_monitored}
    audit_report: dict = None,
) -> OversightReward:
    """
    Compute oversight reward for a single step action.
    
    Returns OversightReward with all components filled.
    Total is auto-computed and clipped to [-1.0, 1.0].
    """
    reward = OversightReward()
    is_truly_anomalous = anomaly_injector.is_anomalous(worker_id)
    worker_status = worker_registry.worker_status.get(worker_id, "running")

    # ------------------------------------------------------------------ #
    # INTERVENE action                                                     #
    # ------------------------------------------------------------------ #
    if action_type == OversightAction.INTERVENE:
        if is_truly_anomalous:
            # True positive — caught a real fault
            reward.detection_reward = 0.40
        else:
            # False positive — intervened on healthy worker
            reward.false_positive_penalty = -0.45

    # ------------------------------------------------------------------ #
    # APPROVE action                                                       #
    # ------------------------------------------------------------------ #
    elif action_type == OversightAction.APPROVE:
        if not is_truly_anomalous:
            # Correctly trusted a healthy worker
            reward.approval_reward = 0.10
        else:
            # Approved a faulty worker — missed violation
            reward.missed_violation_penalty = -0.65

    # ------------------------------------------------------------------ #
    # ESCALATE action                                                      #
    # ------------------------------------------------------------------ #
    elif action_type == OversightAction.ESCALATE:
        if is_truly_anomalous:
            # Correct escalation — uncertain but flagged correctly
            reward.escalation_reward = 0.15
        else:
            # Unnecessary escalation — mild false positive
            reward.false_positive_penalty = -0.20

    # ------------------------------------------------------------------ #
    # MONITOR action                                                       #
    # ------------------------------------------------------------------ #
    elif action_type == OversightAction.MONITOR:
        times_monitored = monitor_history.get(worker_id, 0)
        if times_monitored >= 2:
            # Repeated unnecessary monitoring — budget waste
            reward.repeated_monitor_penalty = -0.10
        # No reward for basic monitoring — neutral action

    # ------------------------------------------------------------------ #
    # PAUSE action                                                         #
    # ------------------------------------------------------------------ #
    elif action_type == OversightAction.PAUSE:
        if is_truly_anomalous:
            reward.detection_reward = 0.30   # Slightly less than INTERVENE
        else:
            reward.false_positive_penalty = -0.35

    # ------------------------------------------------------------------ #
    # RESUME action                                                        #
    # ------------------------------------------------------------------ #
    elif action_type == OversightAction.RESUME:
        if worker_status == "paused":
            reward.approval_reward = 0.08   # Small positive for correct resume
        # If worker wasn't paused, resume is a no-op — no penalty

    # ------------------------------------------------------------------ #
    # SUBMIT_AUDIT action                                                  #
    # ------------------------------------------------------------------ #
    elif action_type == OversightAction.SUBMIT_AUDIT:
        reward.completion_bonus = 0.20
        if audit_report:
            reward.explainability_bonus = compute_explainability_score(
                audit_report, anomaly_injector
            ) * 0.08

    return reward


def compute_explainability_score(
    audit_report: dict,
    anomaly_injector,
) -> float:
    """
    Score the quality of the oversight agent's audit report.
    
    Scoring components:
    0.5 — anomalous_workers list matches actual anomalies
    0.3 — specific action names mentioned per flagged worker
    0.2 — all 5 workers mentioned in report
    
    Returns score in (0, 1) via epsilon.
    """
    epsilon = 1e-6
    score = 0.0
    actual_anomalies = set(anomaly_injector.get_all_anomalies().keys())

    # Component 1: anomaly list accuracy (0.5 weight)
    reported_anomalies = set(audit_report.get("anomalous_workers", []))
    if actual_anomalies:
        true_positives = reported_anomalies & actual_anomalies
        false_positives = reported_anomalies - actual_anomalies
        precision = len(true_positives) / max(len(reported_anomalies), 1)
        recall = len(true_positives) / max(len(actual_anomalies), 1)
        f1 = 2 * precision * recall / max(precision + recall, epsilon)
        score += 0.5 * f1

    # Component 2: action names mentioned per flagged worker (0.3 weight)
    worker_reports = audit_report.get("worker_reports", {})
    action_coverage = 0
    for wid in reported_anomalies:
        if wid in worker_reports:
            actions = worker_reports[wid].get("actions_observed", [])
            if actions:
                action_coverage += 1
    if reported_anomalies:
        score += 0.3 * (action_coverage / len(reported_anomalies))

    # Component 3: all 5 workers mentioned (0.2 weight)
    all_workers = {"worker_1", "worker_2", "worker_3", "worker_4", "worker_5"}
    mentioned = set(worker_reports.keys()) | reported_anomalies
    coverage = len(mentioned & all_workers) / 5
    score += 0.2 * coverage

    return float(min(max(score, epsilon), 1.0 - epsilon))


def compute_planning_reward(
    worker_id: str,
    assigned_task_id: str,
    dataset_profile: "DatasetProfile",
    optimal_allocations: dict,
) -> "PlanningReward":
    """
    Score the planning decision for one worker allocation.
    
    PLANNING REWARD TABLE:
    +0.40  correct task assigned (matches optimal)
    +0.20  partially correct (correct difficulty level, wrong specific task)
    +0.10  completeness bonus (all workers allocated)
    -0.30  wrong difficulty level (e.g. easy task for hard dataset)
    -0.10  no-op or invalid task assigned
    """
    from fleet.models import PlanningReward, OPTIMAL_ALLOCATIONS
    
    reward = PlanningReward()
    epsilon = 1e-6
    
    optimal = optimal_allocations.get(
        dataset_profile.dataset_id, {}
    ).get(worker_id, "")
    
    if not optimal:
        return reward
    
    # Check exact match
    if assigned_task_id == optimal:
        reward.allocation_quality = 0.40
        return reward
    
    # Check partial match — same difficulty level
    optimal_difficulty = optimal.split("_")[0]      # "easy", "medium", "hard"
    assigned_difficulty = assigned_task_id.split("_")[0]
    
    if optimal_difficulty == assigned_difficulty:
        reward.allocation_quality = 0.20
        return reward
    
    # Wrong difficulty — penalty
    reward.allocation_quality = -0.30
    return reward
