"""
Worker Agent 5 — Evaluation Agent
Task: Score the complete RAG pipeline end-to-end.
Measures faithfulness, relevance, and pipeline integrity.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Optional

from workers.base_worker import BaseWorker
from workers.embedding_env import mock_embed

EVALUATION_TASK_CONFIGS = {
    "easy_evaluation": {"task_id": "easy_evaluation", "difficulty": "easy", "step_budget": 8},
    "medium_evaluation": {"task_id": "medium_evaluation", "difficulty": "medium", "step_budget": 10},
    "hard_evaluation": {"task_id": "hard_evaluation", "difficulty": "hard", "step_budget": 12},
}


def cosine_sim(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a)) or 1.0
    nb = math.sqrt(sum(x * x for x in b)) or 1.0
    return dot / (na * nb)


class EvaluationEnv(BaseWorker):
    """
    Worker Agent 5: Evaluation Agent.

    Actions:
    - run_faithfulness_check: Check if retrieved chunks contain expected answers
    - run_relevance_check: Check semantic overlap of retrieved chunks with queries
    - check_pipeline_integrity: Verify all upstream outputs exist
    - compute_composite_score: Compute weighted final score
    - generate_eval_report: Generate full evaluation report
    - submit: Submit evaluation results
    """

    VALID_ACTIONS = [
        "run_faithfulness_check", "run_relevance_check",
        "check_pipeline_integrity", "compute_composite_score",
        "generate_eval_report", "submit"
    ]

    def __init__(self) -> None:
        super().__init__(worker_id="worker_5", worker_name="Evaluation Agent")
        self.faithfulness_score: float = 0.0
        self.relevance_score: float = 0.0
        self.pipeline_integrity_score: float = 0.0
        self.composite_score: float = 0.0
        self.faithfulness_done: bool = False
        self.relevance_done: bool = False
        self.integrity_done: bool = False
        self.composite_done: bool = False
        self.eval_report: dict = {}
        self.task_config: dict = {}
        self.index: dict = {}
        self.retrieval_results: list[dict] = []
        self.ground_truth: list[dict] = []
        self.chunk_data: dict[str, str] = {}      # chunk_id -> text

    def reset(
        self,
        task_id: str,
        index: Optional[dict] = None,
        retrieval_results: Optional[list] = None,
        chunk_data: Optional[dict] = None,
    ) -> dict:
        self._reset_episode_tracking()
        self.task_id = task_id
        self.task_config = EVALUATION_TASK_CONFIGS.get(task_id, EVALUATION_TASK_CONFIGS["easy_evaluation"])
        self.step_budget = self.task_config["step_budget"]
        self.step_budget_remaining = self.step_budget
        self.faithfulness_score = 0.0
        self.relevance_score = 0.0
        self.pipeline_integrity_score = 0.0
        self.composite_score = 0.0
        self.faithfulness_done = False
        self.relevance_done = False
        self.integrity_done = False
        self.composite_done = False
        self.eval_report = {}
        self.index = index or {}
        self.retrieval_results = retrieval_results or []
        self.chunk_data = chunk_data or self._load_chunk_data()
        self.ground_truth = self._load_ground_truth()
        return self.state()

    def step(self, action_dict: dict) -> tuple[dict, float, bool, dict]:
        operation = action_dict.get("operation", "")
        parameters = action_dict.get("parameters", {})

        if self.is_done:
            return self.state(), 0.0, True, {"error": "episode_already_done", "action": operation}

        if operation not in self.VALID_ACTIONS:
            self._record_governance_event("invalid_action", "medium", f"Unknown: {operation}")
            reward = self._clip_reward(0.0)
            self.step_count += 1
            self.step_budget_remaining -= 1
            self.total_reward += reward
            if self._is_budget_exhausted():
                self.is_done = True
            info = {"error": f"invalid_action:{operation}", "action": operation}
            self._record_action(operation, reward, info)
            return self.state(), reward, self.is_done, info

        reward, info = self._dispatch(operation, parameters)
        reward = self._clip_reward(reward)
        self.step_count += 1
        self.step_budget_remaining -= 1
        self.total_reward += reward
        done = self.is_done or self._is_budget_exhausted()
        if done:
            self.is_done = True
        self._record_action(operation, reward, info)
        return self.state(), reward, done, info

    def state(self) -> dict:
        base = self._get_base_state()
        base.update({
            "faithfulness_score": round(self.faithfulness_score, 4),
            "relevance_score": round(self.relevance_score, 4),
            "pipeline_integrity_score": round(self.pipeline_integrity_score, 4),
            "composite_score": round(self.composite_score, 4),
            "faithfulness_done": self.faithfulness_done,
            "relevance_done": self.relevance_done,
            "integrity_done": self.integrity_done,
            "composite_done": self.composite_done,
        })
        return base

    def generate_run_report(self) -> dict:
        report = self._get_base_report()
        report.update({
            "faithfulness_score": self.faithfulness_score,
            "relevance_score": self.relevance_score,
            "pipeline_integrity_score": self.pipeline_integrity_score,
            "composite_score": self.composite_score,
            "eval_report": self.eval_report,
            "final_score": self._compute_final_score(),
        })
        return report

    def evaluate_run(self) -> dict:
        epsilon = 1e-6
        score = min(max(self._compute_final_score(), epsilon), 1.0 - epsilon)
        gates = {
            "faithfulness_checked": self.faithfulness_done,
            "relevance_checked": self.relevance_done,
            "integrity_checked": self.integrity_done,
            "composite_computed": self.composite_done,
            "submitted": self.submitted,
        }
        return {"approved": all(gates.values()) and score >= 0.55, "gates": gates, "composite_score": score}

    def _dispatch(self, operation: str, parameters: dict) -> tuple[float, dict]:
        if operation == "run_faithfulness_check":
            return self._run_faithfulness_check()
        elif operation == "run_relevance_check":
            return self._run_relevance_check()
        elif operation == "check_pipeline_integrity":
            return self._check_pipeline_integrity()
        elif operation == "compute_composite_score":
            return self._compute_composite_score()
        elif operation == "generate_eval_report":
            return self._generate_eval_report()
        elif operation == "submit":
            return self._submit()
        return 0.0, {"error": "unknown_operation"}

    def _run_faithfulness_check(self) -> tuple[float, dict]:
        hits = 0
        for qa in self.ground_truth:
            chunk_text = self.chunk_data.get(qa["chunk_id"], "")
            answer = qa["answer"].lower()
            if answer in chunk_text.lower():
                hits += 1
        self.faithfulness_score = hits / max(len(self.ground_truth), 1)
        self.faithfulness_done = True
        reward = 0.4 * self.faithfulness_score + 0.1
        return reward, {"error": None, "action": "run_faithfulness_check", "faithfulness_score": round(self.faithfulness_score, 4)}

    def _run_relevance_check(self) -> tuple[float, dict]:
        if not self.retrieval_results:
            self.relevance_score = 0.5
            self.relevance_done = True
            return 0.2, {"error": None, "action": "run_relevance_check", "relevance_score": 0.5}
        scores = []
        for result in self.retrieval_results[:10]:
            q_vec = mock_embed(result["question"])
            for cid in result.get("retrieved_chunk_ids", [])[:3]:
                chunk_text = self.chunk_data.get(cid, result["question"])
                c_vec = mock_embed(chunk_text)
                scores.append(cosine_sim(q_vec, c_vec))
        self.relevance_score = sum(scores) / max(len(scores), 1)
        self.relevance_done = True
        reward = 0.3 * self.relevance_score + 0.1
        return reward, {"error": None, "action": "run_relevance_check", "relevance_score": round(self.relevance_score, 4)}

    def _check_pipeline_integrity(self) -> tuple[float, dict]:
        checks = {
            "index_exists": len(self.index) > 0,
            "chunks_exist": len(self.chunk_data) > 0,
            "ground_truth_exists": len(self.ground_truth) > 0,
            "retrieval_results_exist": len(self.retrieval_results) > 0,
        }
        passed = sum(checks.values())
        self.pipeline_integrity_score = passed / len(checks)
        self.integrity_done = True
        reward = 0.2 * self.pipeline_integrity_score + 0.1
        return reward, {"error": None, "action": "check_pipeline_integrity", "checks": checks, "integrity_score": round(self.pipeline_integrity_score, 4)}

    def _compute_composite_score(self) -> tuple[float, dict]:
        epsilon = 1e-6
        raw = (
            0.4 * self.faithfulness_score +
            0.3 * self.relevance_score +
            0.2 * self.pipeline_integrity_score +
            0.1 * (self.step_budget_remaining / max(self.step_budget, 1))
        )
        self.composite_score = min(max(raw, epsilon), 1.0 - epsilon)
        self.composite_done = True
        return 0.3, {"error": None, "action": "compute_composite_score", "composite_score": round(self.composite_score, 4)}

    def _generate_eval_report(self) -> tuple[float, dict]:
        self.eval_report = {
            "faithfulness_score": round(self.faithfulness_score, 4),
            "relevance_score": round(self.relevance_score, 4),
            "pipeline_integrity_score": round(self.pipeline_integrity_score, 4),
            "composite_score": round(self.composite_score, 4),
            "checks_completed": {
                "faithfulness": self.faithfulness_done,
                "relevance": self.relevance_done,
                "integrity": self.integrity_done,
                "composite": self.composite_done,
            },
            "recommendation": "APPROVE" if self.composite_score >= 0.6 else "REJECT",
        }
        return 0.25, {"error": None, "action": "generate_eval_report", "report": self.eval_report}

    def _submit(self) -> tuple[float, dict]:
        if not (self.faithfulness_done and self.relevance_done and self.integrity_done):
            self._record_governance_event("incomplete_eval", "high", "submit before all checks done")
            return 0.05, {"error": "evaluation_incomplete", "action": "submit"}
        self.submitted = True
        self.is_done = True
        final_score = self._compute_final_score()
        return min(final_score + 0.15, 0.99), {"error": None, "action": "submit", "final_score": final_score}

    def _compute_final_score(self) -> float:
        epsilon = 1e-6
        raw = (
            0.4 * self.faithfulness_score +
            0.3 * self.relevance_score +
            0.2 * self.pipeline_integrity_score +
            0.1 * (self.step_budget_remaining / max(self.step_budget, 1))
        )
        return float(min(max(raw, epsilon), 1.0 - epsilon))

    def _load_chunk_data(self) -> dict[str, str]:
        path = Path("data/nexacrm_corpus.json")
        if path.exists():
            with open(path) as f:
                corpus = json.load(f)
            return {c["chunk_id"]: c["text"] for c in corpus}
        from data.setup_dataset import GROUND_TRUTH_QA
        return {qa["chunk_id"]: f"{qa['question']} {qa['answer']}" for qa in GROUND_TRUTH_QA}

    def _load_ground_truth(self) -> list[dict]:
        path = Path("data/ground_truth_qa.json")
        if path.exists():
            with open(path) as f:
                return json.load(f)[:20]
        from data.setup_dataset import GROUND_TRUTH_QA
        return GROUND_TRUTH_QA[:20]
