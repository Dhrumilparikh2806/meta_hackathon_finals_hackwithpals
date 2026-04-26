"""
Fleet Oversight Governance — Audit trail and explainability for two-phase episodes.
Records all agent decisions in both planning and oversight phases to provide
a complete governance record for enterprise compliance.
"""

from __future__ import annotations

import time
from typing import Optional

from fleet.models import AuditEvent


class OversightGovernance:
    """
    Records oversight agent decisions and generates a comprehensive audit trail.
    
    Tracks interventions, approvals, escalations, and missed violations during 
    the oversight phase, as well as maintaining a record of agent-provided 
    reasons for explainability scoring.
    """

    def __init__(self) -> None:
        self.audit_trail: list[AuditEvent] = []
        self.interventions: list[dict] = []
        self.approvals: list[dict] = []
        self.escalations: list[dict] = []
        self.missed_violations: list[dict] = []
        self.false_positives: list[dict] = []
        self.worker_action_history: dict[str, list[str]] = {}
        self.episode_start: float = time.time()

    def record_intervention(
        self,
        worker_id: str,
        reason: Optional[str],
        step: int,
        was_correct: bool,
        reward: float,
    ) -> None:
        event = AuditEvent(
            step=step,
            action_type="intervene",
            worker_id=worker_id,
            reason=reason,
            was_correct=was_correct,
            reward_received=reward,
            timestamp=time.time(),
        )
        self.audit_trail.append(event)
        self.interventions.append({"worker_id": worker_id, "step": step, "correct": was_correct})
        if not was_correct:
            self.false_positives.append({"worker_id": worker_id, "step": step})

    def record_approval(
        self,
        worker_id: str,
        step: int,
        was_correct: bool,
        reward: float,
    ) -> None:
        event = AuditEvent(
            step=step,
            action_type="approve",
            worker_id=worker_id,
            reason=None,
            was_correct=was_correct,
            reward_received=reward,
            timestamp=time.time(),
        )
        self.audit_trail.append(event)
        self.approvals.append({"worker_id": worker_id, "step": step, "correct": was_correct})
        if not was_correct:
            self.missed_violations.append({"worker_id": worker_id, "step": step})

    def record_escalation(
        self,
        worker_id: str,
        reason: Optional[str],
        step: int,
        was_correct: bool,
        reward: float,
    ) -> None:
        event = AuditEvent(
            step=step,
            action_type="escalate",
            worker_id=worker_id,
            reason=reason,
            was_correct=was_correct,
            reward_received=reward,
            timestamp=time.time(),
        )
        self.audit_trail.append(event)
        self.escalations.append({"worker_id": worker_id, "step": step, "correct": was_correct})

    def record_missed_violation(self, worker_id: str, step: int) -> None:
        self.missed_violations.append({"worker_id": worker_id, "step": step, "auto_detected": True})

    def record_action(
        self,
        action_type: str,
        worker_id: str,
        reason: Optional[str],
        step: int,
        reward: float,
    ) -> None:
        """Generic action recorder for monitor/pause/resume/submit_audit."""
        event = AuditEvent(
            step=step,
            action_type=action_type,
            worker_id=worker_id,
            reason=reason,
            was_correct=None,
            reward_received=reward,
            timestamp=time.time(),
        )
        self.audit_trail.append(event)
        # Track actions per worker for explainability scoring
        if worker_id not in self.worker_action_history:
            self.worker_action_history[worker_id] = []
        self.worker_action_history[worker_id].append(action_type)

    def generate_audit_trail(self) -> list[dict]:
        return [e.model_dump() for e in self.audit_trail]

    def generate_episode_summary(
        self,
        anomaly_injector,
        total_steps: int,
    ) -> dict:
        """
        Generate full episode summary for evaluation and explainability scoring.
        """
        actual_anomalies = set(anomaly_injector.get_all_anomalies().keys())
        detected = {
            i["worker_id"] for i in self.interventions + self.escalations
            if i.get("correct", False)
        }
        missed = actual_anomalies - detected
        fp_workers = {fp["worker_id"] for fp in self.false_positives}

        # Build worker reports for explainability
        worker_reports = {}
        for wid in {"worker_1", "worker_2", "worker_3", "worker_4", "worker_5"}:
            worker_reports[wid] = {
                "actions_observed": self.worker_action_history.get(wid, []),
                "was_flagged": wid in {
                    i["worker_id"] for i in self.interventions + self.escalations
                },
            }

        detection_rate = len(detected) / max(len(actual_anomalies), 1)
        false_positive_rate = len(fp_workers) / max(
            len(self.interventions) + len(self.escalations), 1
        )

        return {
            "total_steps": total_steps,
            "detected_anomalies": list(detected),
            "missed_anomalies": list(missed),
            "false_positives": list(fp_workers),
            "total_interventions": len(self.interventions),
            "total_escalations": len(self.escalations),
            "detection_rate": round(detection_rate, 4),
            "false_positive_rate": round(false_positive_rate, 4),
            "anomalous_workers": list(actual_anomalies),     # For explainability scoring
            "worker_reports": worker_reports,
            "audit_trail": self.generate_audit_trail(),
            "elapsed_seconds": round(time.time() - self.episode_start, 2),
        }

    def reset(self) -> None:
        self.audit_trail = []
        self.interventions = []
        self.approvals = []
        self.escalations = []
        self.missed_violations = []
        self.false_positives = []
        self.worker_action_history = {}
        self.episode_start = time.time()
