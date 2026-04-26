"""
Worker Agent 4 — Retrieval Agent
Task: Configure retrieval system and validate against ground truth QA pairs.
Reward based on precision@3 against hardcoded ground truth.
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Optional

from workers.base_worker import BaseWorker
from workers.embedding_env import mock_embed

RETRIEVAL_TASK_CONFIGS = {
    "easy_retrieval": {"task_id": "easy_retrieval", "difficulty": "easy", "step_budget": 8, "n_eval_questions": 10},
    "medium_retrieval": {"task_id": "medium_retrieval", "difficulty": "medium", "step_budget": 10, "n_eval_questions": 25},
    "hard_retrieval": {"task_id": "hard_retrieval", "difficulty": "hard", "step_budget": 12, "n_eval_questions": 50},
}

VALID_RERANKERS = ["none", "bm25", "cross_encoder", "reciprocal_rank_fusion"]


def cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a)) or 1.0
    norm_b = math.sqrt(sum(x * x for x in b)) or 1.0
    return dot / (norm_a * norm_b)


class RetrievalEnv(BaseWorker):
    """
    Worker Agent 4: Retrieval Agent.

    Actions:
    - configure_retrieval: Set top_k value
    - set_reranker: Choose reranking strategy
    - preprocess_query: Clean and normalize a query
    - run_retrieval: Run retrieval on eval questions
    - evaluate_precision: Score retrieval against ground truth
    - submit: Submit retrieval config and scores
    """

    VALID_ACTIONS = [
        "configure_retrieval", "set_reranker", "preprocess_query",
        "run_retrieval", "evaluate_precision", "submit"
    ]

    def __init__(self) -> None:
        super().__init__(worker_id="worker_4", worker_name="Retrieval Agent")
        self.top_k: int = 5
        self.reranker: str = "none"
        self.queries_tested: int = 0
        self.precision_at_3: float = 0.0
        self.retrieval_done: bool = False
        self.task_config: dict = {}
        self.index: dict[str, list[float]] = {}
        self.ground_truth: list[dict] = []
        self.retrieval_results: list[dict] = []

    def reset(self, task_id: str, index: Optional[dict] = None) -> dict:
        self._reset_episode_tracking()
        self.task_id = task_id
        self.task_config = RETRIEVAL_TASK_CONFIGS.get(task_id, RETRIEVAL_TASK_CONFIGS["easy_retrieval"])
        self.step_budget = self.task_config["step_budget"]
        self.step_budget_remaining = self.step_budget
        self.top_k = 5
        self.reranker = "none"
        self.queries_tested = 0
        self.precision_at_3 = 0.0
        self.retrieval_done = False
        self.retrieval_results = []
        self.index = index or self._build_mock_index()
        self.ground_truth = self._load_ground_truth(self.task_config["n_eval_questions"])
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
            done = self._is_budget_exhausted()
            if done:
                self.is_done = True
            info = {"error": f"invalid_action:{operation}", "action": operation}
            self._record_action(operation, reward, info)
            return self.state(), reward, done, info

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
            "top_k": self.top_k,
            "reranker": self.reranker,
            "queries_tested": self.queries_tested,
            "precision_at_3": round(self.precision_at_3, 4),
            "retrieval_done": self.retrieval_done,
            "index_size": len(self.index),
            "ground_truth_count": len(self.ground_truth),
        })
        return base

    def generate_run_report(self) -> dict:
        report = self._get_base_report()
        report.update({
            "top_k": self.top_k,
            "reranker": self.reranker,
            "precision_at_3": self.precision_at_3,
            "queries_tested": self.queries_tested,
            "final_score": self._compute_final_score(),
        })
        return report

    def evaluate_run(self) -> dict:
        epsilon = 1e-6
        score = min(max(self._compute_final_score(), epsilon), 1.0 - epsilon)
        gates = {
            "retrieval_configured": self.top_k > 0,
            "retrieval_executed": self.retrieval_done,
            "precision_acceptable": self.precision_at_3 >= 0.3,
            "submitted": self.submitted,
        }
        return {"approved": all(gates.values()) and score >= 0.5, "gates": gates, "composite_score": score}

    def _dispatch(self, operation: str, parameters: dict) -> tuple[float, dict]:
        if operation == "configure_retrieval":
            return self._configure_retrieval(parameters)
        elif operation == "set_reranker":
            return self._set_reranker(parameters)
        elif operation == "preprocess_query":
            return self._preprocess_query(parameters)
        elif operation == "run_retrieval":
            return self._run_retrieval()
        elif operation == "evaluate_precision":
            return self._evaluate_precision()
        elif operation == "submit":
            return self._submit()
        return 0.0, {"error": "unknown_operation"}

    def _configure_retrieval(self, params: dict) -> tuple[float, dict]:
        try:
            k = int(params.get("top_k", 5))
            self.top_k = max(1, min(k, 20))
            reward = 0.4 if 3 <= self.top_k <= 7 else 0.2
            return reward, {"error": None, "action": "configure_retrieval", "top_k": self.top_k}
        except (ValueError, TypeError):
            return 0.0, {"error": "invalid_top_k", "action": "configure_retrieval"}

    def _set_reranker(self, params: dict) -> tuple[float, dict]:
        reranker = params.get("strategy", "none")
        if reranker not in VALID_RERANKERS:
            return 0.1, {"error": f"invalid_reranker:{reranker}", "action": "set_reranker"}
        self.reranker = reranker
        reward = 0.3 if reranker in ("bm25", "reciprocal_rank_fusion") else 0.15
        return reward, {"error": None, "action": "set_reranker", "reranker": reranker}

    def _preprocess_query(self, params: dict) -> tuple[float, dict]:
        query = params.get("query", "")
        processed = query.lower().strip()
        return 0.2, {"error": None, "action": "preprocess_query", "processed": processed}

    def _run_retrieval(self) -> tuple[float, dict]:
        if not self.index:
            self._record_governance_event("no_index", "high", "run_retrieval called with empty index")
            return 0.05, {"error": "empty_index", "action": "run_retrieval"}

        self.retrieval_results = []
        for qa in self.ground_truth:
            query_vec = mock_embed(qa["question"])
            scores = {
                cid: cosine_similarity(query_vec, vec)
                for cid, vec in self.index.items()
            }
            top_chunks = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:self.top_k]
            self.retrieval_results.append({
                "qa_id": qa["id"],
                "question": qa["question"],
                "correct_chunk_id": qa["chunk_id"],
                "retrieved_chunk_ids": [c[0] for c in top_chunks],
                "scores": [round(c[1], 4) for c in top_chunks],
            })

        self.queries_tested = len(self.retrieval_results)
        self.retrieval_done = True
        hits = sum(
            1 for r in self.retrieval_results
            if r["correct_chunk_id"] in r["retrieved_chunk_ids"][:3]
        )
        self.precision_at_3 = hits / max(self.queries_tested, 1)
        reward = 0.5 * self.precision_at_3 + 0.2 * (self.step_budget_remaining / max(self.step_budget, 1))
        return reward, {"error": None, "action": "run_retrieval", "precision_at_3": round(self.precision_at_3, 4)}

    def _evaluate_precision(self) -> tuple[float, dict]:
        if not self.retrieval_done:
            return 0.1, {"error": "retrieval_not_done", "action": "evaluate_precision"}
        report = {
            "precision_at_3": round(self.precision_at_3, 4),
            "queries_tested": self.queries_tested,
            "hits": int(self.precision_at_3 * self.queries_tested),
        }
        reward = 0.4 if self.precision_at_3 >= 0.5 else 0.2
        return reward, {"error": None, "action": "evaluate_precision", "report": report}

    def _submit(self) -> tuple[float, dict]:
        if not self.retrieval_done:
            self._record_governance_event("premature_submit", "high", "submit before run_retrieval")
            return 0.05, {"error": "retrieval_not_done", "action": "submit"}
        self.submitted = True
        self.is_done = True
        final_score = self._compute_final_score()
        return min(final_score + 0.15, 0.99), {"error": None, "action": "submit", "final_score": final_score}

    def _compute_final_score(self) -> float:
        if not self.retrieval_done:
            return 0.001
        score = (
            0.5 * self.precision_at_3 +
            0.3 * (1.0 if self.reranker != "none" else 0.5) +
            0.2 * (self.step_budget_remaining / max(self.step_budget, 1))
        )
        return self._clip_reward(score)

    def _build_mock_index(self) -> dict[str, list[float]]:
        path = Path("data/nexacrm_corpus.json")
        if path.exists():
            with open(path) as f:
                corpus = json.load(f)
            return {c["chunk_id"]: mock_embed(c["text"]) for c in corpus}
        return {f"chunk_{i:03d}": mock_embed(f"NexaCRM entry {i}") for i in range(100)}

    def _load_ground_truth(self, n: int) -> list[dict]:
        path = Path("data/ground_truth_qa.json")
        if path.exists():
            with open(path) as f:
                return json.load(f)[:n]
        from data.setup_dataset import GROUND_TRUTH_QA
        return GROUND_TRUTH_QA[:n]
