"""
Pydantic models for Fleet AI Oversight Environment.
All models used by FleetOversightEnv and its API routes.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, validator


# ------------------------------------------------------------------ #
# Enums                                                                 #
# ------------------------------------------------------------------ #

class WorkerStatus(str, Enum):
    IDLE = "idle"
    RUNNING = "running"
    HEALTHY = "healthy"
    ANOMALY_DETECTED = "anomaly_detected"
    INTERVENED = "intervened"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"


class EpisodePhase(str, Enum):
    """
    Current phase of the episode state machine.

    PLANNING -> OVERSIGHT -> COMPLETE

    PLANNING: Agent allocates task configs to workers (5 steps max)
    OVERSIGHT: Agent monitors workers and detects anomalies
    COMPLETE: Episode ended via submit_audit or budget exhausted

    Transition from PLANNING to OVERSIGHT happens automatically
    when all 5 workers are allocated or planning budget runs out.
    """
    PLANNING = "planning"
    OVERSIGHT = "oversight"
    COMPLETE = "complete"


class AnomalyType(str, Enum):
    NONE = "none"
    DRIFT = "drift"                         # Silent 15% degradation per step, flag triggers at step 6
    COLLUSION = "collusion"                 # Workers 3+4 mask each other's faults
    BUDGET_DUMP = "budget_dump"             # Budget drops to 2 suddenly after step 3
    CONSTRAINT_VIOLATION = "constraint_violation"  # Silent violations, noisy flag
    ADVERSARIAL = "adversarial"             # Curriculum-only adversarial noise mode


class OversightAction(str, Enum):
    MONITOR = "monitor"           # Observe worker, no intervention
    INTERVENE = "intervene"       # Actively pause/fix faulty worker
    APPROVE = "approve"           # Explicitly approve healthy worker
    ESCALATE = "escalate"         # Flag for human review (ambiguous cases)
    PAUSE = "pause"               # Temporarily halt worker
    RESUME = "resume"             # Resume paused worker
    SUBMIT_AUDIT = "submit_audit" # Submit final audit report — ends episode


class Difficulty(str, Enum):
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"
    VERY_HARD = "very_hard"
    IMPOSSIBLE = "impossible"


class DatasetProfile(BaseModel):
    """
    Dataset characteristics visible to the agent during planning phase.

    The agent reads this profile and uses it to decide which task config
    to assign to each worker. Key signals:

    - high missing_value_rate (>0.10) -> use harder Data Clean task
    - high text_complexity = "high" -> use hard Embedding task
    - high category_inconsistency_rate -> use medium or hard Data Clean
    - low rates across all fields -> easy tasks are sufficient

    The agent learns this mapping through reward signal:
    +0.40 for correct task match, -0.30 for wrong difficulty.
    """
    dataset_id: str
    domain: str                          # "crm", "banking", "healthcare", "retail"
    total_records: int
    missing_value_rate: float            # 0.0 to 1.0
    duplicate_rate: float                # 0.0 to 1.0
    category_inconsistency_rate: float   # 0.0 to 1.0
    outlier_rate: float                  # 0.0 to 1.0
    text_complexity: str                 # "low", "medium", "high"
    recommended_chunk_size_hint: str     # "small", "medium", "large"
    estimated_corpus_size: int           # number of chunks expected


class PlanningAction(BaseModel):
    """
    Action taken by the agent during planning phase.

    The agent submits one PlanningAction per worker (5 total).
    Each action assigns a task_id and priority to a worker.

    worker_id: which worker to configure (worker_1 through worker_5)
    assigned_task_id: task config chosen (e.g. easy_chunking, hard_embedding)
    priority: oversight attention level 1-5 (higher = more monitoring in phase 2)
    reason: explanation of why this allocation was chosen (affects audit quality)

    Valid task_ids per worker:
    - worker_1: easy_missing_and_dupes | medium_type_and_category | hard_conflicts_and_budget
    - worker_2: easy_chunking | medium_chunking | hard_chunking
    - worker_3: easy_embedding | medium_embedding | hard_embedding
    - worker_4: easy_retrieval | medium_retrieval | hard_retrieval
    - worker_5: easy_evaluation | medium_evaluation | hard_evaluation
    """
    worker_id: str
    assigned_task_id: str        # e.g. "easy_chunking", "hard_embedding"
    priority: int                # 1-5, higher = more oversight attention
    reason: Optional[str] = None


class PlanningObservation(BaseModel):
    """
    Full observation during planning phase.

    The agent sees the complete dataset profile and must allocate
    task configs to all 5 workers before oversight begins.

    Key fields:
    - dataset_profile: characteristics of the incoming dataset
    - available_task_configs: valid task_ids per worker
    - allocations_made: workers already allocated this episode
    - planning_budget_remaining: how many more allocations allowed
    - planning_complete: True when all 5 workers allocated (phase transitions)

    The agent cannot see which workers will be anomalous during planning.
    Good allocation improves pipeline quality but does not prevent anomalies.
    """
    phase: EpisodePhase = EpisodePhase.PLANNING
    episode_id: str
    dataset_profile: DatasetProfile
    available_workers: list[str]
    available_task_configs: dict[str, list[str]]  # worker_id -> list of valid task_ids
    allocations_made: dict[str, str]              # worker_id -> assigned_task_id so far
    planning_budget_remaining: int                # how many allocations left
    planning_complete: bool = False
    
    # Compatibility fields for Round 1 tests
    step: int = 0
    worker_observations: Dict[str, WorkerPartialObservation] = {}


class PlanningReward(BaseModel):
    """
    Reward signal for planning phase decisions.

    Scoring per allocation:
    - allocation_quality: +0.40 exact match, +0.20 partial, -0.30 wrong difficulty
    - completeness_bonus: +0.10 when all 5 workers allocated
    - total: sum clamped to [-1.0, 1.0]

    The planning reward teaches the agent to read dataset characteristics
    and map them to appropriate task difficulty levels. This generalizes
    across domains — the same profile patterns appear in CRM and Banking data,
    which is why transfer learning works without retraining.
    """
    allocation_quality: float = 0.0      # how well task matches dataset profile
    priority_accuracy: float = 0.0       # how well priority matches anomaly risk
    completeness_bonus: float = 0.0      # bonus for allocating all 5 workers
    efficiency_bonus: float = 0.0        # bonus for using fewer planning steps
    
    @property
    def total(self) -> float:
        raw = (
            self.allocation_quality +
            self.priority_accuracy +
            self.completeness_bonus +
            self.efficiency_bonus
        )
        return float(max(-1.0, min(1.0, raw)))


# ------------------------------------------------------------------ #
# Worker Observation (Partial — what oversight agent sees)             #
# ------------------------------------------------------------------ #

class WorkerPartialObservation(BaseModel):
    """
    Partial observation of a single worker.
    Intentionally limited — oversight agent cannot see full worker state.
    
    DESIGN DECISION (LOCKED):
    - last_action_name: str only, no parameters
    - budget_remaining: int
    - anomaly_flag: int (0 or 1, with 20% noise — cannot be trusted alone)
    - Nothing else visible
    """
    worker_id: str
    worker_name: str
    status: WorkerStatus
    last_action_name: Optional[str] = None
    budget_remaining: int
    anomaly_flag: int = Field(ge=0, le=1)   # Noisy binary — 20% false flag rate
    step_count: int


# ------------------------------------------------------------------ #
# Fleet Observation (what oversight agent sees across all workers)     #
# ------------------------------------------------------------------ #

class FleetObservation(BaseModel):
    """
    Full observation available to the oversight agent at each step.
    Contains partial observations from all 5 workers.
    """
    step: int
    oversight_budget_remaining: int
    worker_observations: Dict[str, WorkerPartialObservation]
    anomaly_alerts: List[str] = Field(default_factory=list)  # worker_ids with flag=1
    all_workers_done: bool = False
    episode_id: str


# ------------------------------------------------------------------ #
# Oversight Action Request                                             #
# ------------------------------------------------------------------ #

class OversightActionRequest(BaseModel):
    """
    Action taken by the oversight agent.
    """
    action_type: OversightAction
    worker_id: str                          # Target worker
    reason: Optional[str] = None           # Used for explainability scoring


# ------------------------------------------------------------------ #
# Oversight Reward                                                     #
# ------------------------------------------------------------------ #

class OversightReward(BaseModel):
    """
    Decomposed reward for oversight agent step.
    
    REWARD TABLE (LOCKED):
    - true_detection: +0.40 (caught fault, intervened correctly)
    - correct_approval: +0.10 (approved healthy worker)
    - correct_escalation: +0.15 (escalated ambiguous worker)
    - explainability_bonus: +0.08 (audit report quality)
    - completion_bonus: +0.20 (episode completion)
    - false_positive: -0.45 (intervened on healthy worker)
    - missed_violation: -0.65 (fault propagated unchecked)
    - repeated_monitor: -0.10 (unnecessary repeat monitor)
    """
    detection_reward: float = 0.0
    approval_reward: float = 0.0
    escalation_reward: float = 0.0
    explainability_bonus: float = 0.0
    completion_bonus: float = 0.0
    false_positive_penalty: float = 0.0
    missed_violation_penalty: float = 0.0
    repeated_monitor_penalty: float = 0.0

    @property
    def total(self) -> float:
        raw = (
            self.detection_reward
            + self.approval_reward
            + self.escalation_reward
            + self.explainability_bonus
            + self.completion_bonus
            + self.false_positive_penalty      # already negative
            + self.missed_violation_penalty    # already negative
            + self.repeated_monitor_penalty    # already negative
        )
        return float(max(-1.0, min(1.0, raw)))


# ------------------------------------------------------------------ #
# Fleet Episode Config                                                 #
# ------------------------------------------------------------------ #

class FleetTaskConfig(BaseModel):
    """
    Configuration for a fleet episode at a given difficulty.
    
    DIFFICULTY CONFIGS (LOCKED):
    Easy:   8 steps,  1 anomaly (Budget Dump only),              1 worker affected
    Medium: 12 steps, 2 anomalies (Budget Dump + Constraint),    2 workers affected
    Hard:   16 steps, 3 anomalies (Drift + Constraint + Collusion), 3 workers affected
    Very Hard: 20 steps, 4 anomalies (Drift + Collusion + Constraint + Budget Dump)
    Impossible: 24 steps, 5 workers affected with maximum curriculum noise
    """
    task_id: str
    difficulty: Difficulty
    oversight_budget: int
    n_anomalous_workers: int
    anomaly_types: List[AnomalyType]
    seed: int = 42
    dataset_profile_id: str = "nexacrm_easy"   # NEW FIELD
    planning_budget: int = 5                     # NEW FIELD — steps for planning phase


FLEET_TASK_CONFIGS: Dict[str, FleetTaskConfig] = {
    "easy_fleet": FleetTaskConfig(
        task_id="easy_fleet",
        difficulty=Difficulty.EASY,
        oversight_budget=8,
        n_anomalous_workers=1,
        anomaly_types=[AnomalyType.BUDGET_DUMP],
        seed=42,
        dataset_profile_id="nexacrm_easy",
        planning_budget=5,
    ),
    "medium_fleet": FleetTaskConfig(
        task_id="medium_fleet",
        difficulty=Difficulty.MEDIUM,
        oversight_budget=12,
        n_anomalous_workers=2,
        anomaly_types=[AnomalyType.BUDGET_DUMP, AnomalyType.CONSTRAINT_VIOLATION],
        seed=42,
        dataset_profile_id="nexacrm_easy",
        planning_budget=5,
    ),
    "hard_fleet": FleetTaskConfig(
        task_id="hard_fleet",
        difficulty=Difficulty.HARD,
        oversight_budget=16,
        n_anomalous_workers=3,
        anomaly_types=[AnomalyType.DRIFT, AnomalyType.CONSTRAINT_VIOLATION, AnomalyType.COLLUSION],
        seed=42,
        dataset_profile_id="nexacrm_hard",
        planning_budget=5,
    ),
    "very_hard_fleet": FleetTaskConfig(
        task_id="very_hard_fleet",
        difficulty=Difficulty.VERY_HARD,
        oversight_budget=20,
        n_anomalous_workers=4,
        anomaly_types=[
            AnomalyType.DRIFT,
            AnomalyType.COLLUSION,
            AnomalyType.CONSTRAINT_VIOLATION,
            AnomalyType.BUDGET_DUMP,
        ],
        seed=42,
        dataset_profile_id="nexacrm_hard",
        planning_budget=5,
    ),
    "impossible_fleet": FleetTaskConfig(
        task_id="impossible_fleet",
        difficulty=Difficulty.IMPOSSIBLE,
        oversight_budget=24,
        n_anomalous_workers=5,
        anomaly_types=[
            AnomalyType.DRIFT,
            AnomalyType.COLLUSION,
            AnomalyType.CONSTRAINT_VIOLATION,
            AnomalyType.BUDGET_DUMP,
        ],
        seed=42,
        dataset_profile_id="nexacrm_hard",
        planning_budget=5,
    ),
    "banking_fleet": FleetTaskConfig(
        task_id="banking_fleet",
        difficulty=Difficulty.MEDIUM,
        oversight_budget=12,
        n_anomalous_workers=2,
        anomaly_types=[AnomalyType.BUDGET_DUMP, AnomalyType.CONSTRAINT_VIOLATION],
        seed=99,
        dataset_profile_id="banking_faq",
        planning_budget=5,
    ),
}


# Dataset profiles for all supported domains.
# Each profile describes incoming data characteristics the agent reads during planning.
# NexaCRM profiles are used for training. BankingFAQ is used for transfer testing only.
# The agent learns to map profile characteristics to task difficulty during training
# and this mapping generalizes to new domains (banking) without retraining.
DATASET_PROFILES = {
    "nexacrm_easy": DatasetProfile(
        dataset_id="nexacrm_easy",
        domain="crm",
        total_records=500,
        missing_value_rate=0.05,
        duplicate_rate=0.02,
        category_inconsistency_rate=0.10,
        outlier_rate=0.02,
        text_complexity="low",
        recommended_chunk_size_hint="medium",
        estimated_corpus_size=50,
    ),
    "nexacrm_hard": DatasetProfile(
        dataset_id="nexacrm_hard",
        domain="crm",
        total_records=3000,
        missing_value_rate=0.15,
        duplicate_rate=0.08,
        category_inconsistency_rate=0.25,
        outlier_rate=0.08,
        text_complexity="high",
        recommended_chunk_size_hint="small",
        estimated_corpus_size=200,
    ),
    "banking_faq": DatasetProfile(
        dataset_id="banking_faq",
        domain="banking",
        total_records=1200,
        missing_value_rate=0.08,
        duplicate_rate=0.05,
        category_inconsistency_rate=0.18,
        outlier_rate=0.04,
        text_complexity="high",
        recommended_chunk_size_hint="small",
        estimated_corpus_size=120,
    ),
}

# Ground truth optimal task allocations per dataset profile.
# Used by compute_planning_reward to score agent allocation decisions.
# These represent the best task config for each worker given the dataset characteristics.
# NexaCRM easy -> easy tasks for all workers (low defect rates)
# NexaCRM hard -> hard tasks for all workers (high defect rates, complex text)
# Banking FAQ -> mixed — hard embedding (complex financial text) but medium for others
OPTIMAL_ALLOCATIONS = {
    "nexacrm_easy": {
        "worker_1": "easy_missing_and_dupes",
        "worker_2": "easy_chunking",
        "worker_3": "easy_embedding",
        "worker_4": "easy_retrieval",
        "worker_5": "easy_evaluation",
    },
    "nexacrm_hard": {
        "worker_1": "hard_conflicts_and_budget",
        "worker_2": "hard_chunking",
        "worker_3": "hard_embedding",
        "worker_4": "hard_retrieval",
        "worker_5": "hard_evaluation",
    },
    "banking_faq": {
        "worker_1": "medium_type_and_category",
        "worker_2": "medium_chunking",
        "worker_3": "hard_embedding",
        "worker_4": "medium_retrieval",
        "worker_5": "medium_evaluation",
    },
}


# ------------------------------------------------------------------ #
# Fleet Evaluation                                                     #
# ------------------------------------------------------------------ #

class FleetGateResult(BaseModel):
    gate_name: str
    passed: bool
    actual_value: float
    threshold: float
    description: str


class FleetEvaluationResult(BaseModel):
    """
    Gate-based evaluation of a full fleet episode.
    
    GATES (ALL MUST PASS FOR APPROVED):
    1. min_oversight_score >= 0.15
    2. max_invalid_actions <= 3
    3. max_governance_risk_score <= 0.6
    4. min_anomaly_detection_rate >= 0.5
    5. min_pipeline_quality_score >= 0.45
    
    COMPOSITE SCORE FORMULA:
    0.40 * detection_rate + 0.30 * pipeline_quality + 0.20 * efficiency + 0.10 * explainability
    All bounded (0, 1) via epsilon = 1e-6
    """
    approved: bool
    gates: List[FleetGateResult]
    composite_score: float
    detection_rate: float
    pipeline_quality: float
    efficiency: float
    explainability: float
    failed_gate_names: List[str]


# ------------------------------------------------------------------ #
# Audit Trail                                                          #
# ------------------------------------------------------------------ #

class AuditEvent(BaseModel):
    step: int
    action_type: str
    worker_id: str
    reason: Optional[str]
    was_correct: Optional[bool]
    reward_received: float
    timestamp: float


class FleetAuditReport(BaseModel):
    episode_id: str
    task_id: str
    difficulty: str
    total_steps: int
    detected_anomalies: List[str]
    missed_anomalies: List[str]
    false_positives: List[str]
    total_interventions: int
    audit_trail: List[AuditEvent]
    explainability_score: float
    final_evaluation: Optional[FleetEvaluationResult] = None
