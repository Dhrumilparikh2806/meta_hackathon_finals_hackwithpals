"""
Worker Agent 3 — Embedding Agent
Task: Embed chunked documents into a vector store using selected model.
Uses mock embeddings for determinism during training.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Optional

from workers.base_worker import BaseWorker

EMBEDDING_TASK_CONFIGS = {
    "easy_embedding": {"task_id": "easy_embedding", "difficulty": "easy", "step_budget": 8, "corpus_size": 50},
    "medium_embedding": {"task_id": "medium_embedding", "difficulty": "medium", "step_budget": 10, "corpus_size": 100},
    "hard_embedding": {"task_id": "hard_embedding", "difficulty": "hard", "step_budget": 12, "corpus_size": 200},
}

VALID_MODELS = ["all-MiniLM-L6-v2", "all-mpnet-base-v2", "paraphrase-multilingual-MiniLM-L12-v2"]
EMBEDDING_DIM = 384


STOP_WORDS = {"does", "is", "a", "the", "does", "support", "nexacrm", "faq", "what", "are", "you", "built", "on", "with", "how", "do", "i", "can", "an", "available", "on", "for", "in", "at", "rest"}

def mock_embed(text: str) -> list[float]:
    """
    Improved keyword-overlap embedding with stop-word filtering.
    Prioritizes unique terms like 'Slack', 'Pricing', or 'GDPR'.
    """
    # Tokenization and filtering
    raw_words = text.lower().replace("?", "").replace("!", "").replace(".", "").replace(",", "").replace("\u2014", " ").split()
    words = [w for w in raw_words if w not in STOP_WORDS and len(w) > 1]
    
    if not words:
        # Fallback to all words if everything was a stopword
        words = [w for w in raw_words if len(w) > 1]
    
    if not words:
        return [0.0] * EMBEDDING_DIM
        
    vec = [0.0] * EMBEDDING_DIM
    for w in words:
        idx = int(hashlib.md5(w.encode()).hexdigest(), 16) % EMBEDDING_DIM
        # Keywords get higher weight
        vec[idx] += 1.0
    
    norm = math.sqrt(sum(x * x for x in vec)) or 1.0
    return [x / norm for x in vec]


class EmbeddingEnv(BaseWorker):
    """
    Worker Agent 3: Embedding Agent.

    Actions:
    - select_model: Choose embedding model
    - configure_batch_size: Set batch size for embedding
    - run_embedding: Embed all chunks
    - validate_coverage: Check coverage ratio
    - inspect_vectors: Sample 3 vectors
    - store_index: Persist index to memory
    - submit: Submit index and receive terminal reward
    """

    VALID_ACTIONS = [
        "select_model", "configure_batch_size", "run_embedding",
        "validate_coverage", "inspect_vectors", "store_index", "submit"
    ]

    def __init__(self) -> None:
        super().__init__(worker_id="worker_3", worker_name="Embedding Agent")
        self.selected_model: Optional[str] = None
        self.batch_size: int = 32
        self.index: dict[str, list[float]] = {}      # chunk_id -> vector
        self.chunks_embedded: int = 0
        self.null_embedding_count: int = 0
        self.coverage_ratio: float = 0.0
        self.index_stored: bool = False
        self.embedding_done: bool = False
        self.task_config: dict = {}
        self.input_chunks: list[dict] = []

    def reset(self, task_id: str, input_chunks: Optional[list[dict]] = None) -> dict:
        self._reset_episode_tracking()
        self.task_id = task_id
        self.task_config = EMBEDDING_TASK_CONFIGS.get(task_id, EMBEDDING_TASK_CONFIGS["easy_embedding"])
        self.step_budget = self.task_config["step_budget"]
        self.step_budget_remaining = self.step_budget
        self.selected_model = None
        self.batch_size = 32
        self.index = {}
        self.chunks_embedded = 0
        self.null_embedding_count = 0
        self.coverage_ratio = 0.0
        self.index_stored = False
        self.embedding_done = False
        self.input_chunks = input_chunks or self._load_chunks(self.task_config["corpus_size"])
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
            "selected_model": self.selected_model,
            "batch_size": self.batch_size,
            "chunks_embedded": self.chunks_embedded,
            "null_embedding_count": self.null_embedding_count,
            "coverage_ratio": round(self.coverage_ratio, 4),
            "index_stored": self.index_stored,
            "embedding_done": self.embedding_done,
            "input_chunk_count": len(self.input_chunks),
        })
        return base

    def generate_run_report(self) -> dict:
        report = self._get_base_report()
        report.update({
            "selected_model": self.selected_model,
            "chunks_embedded": self.chunks_embedded,
            "null_embedding_count": self.null_embedding_count,
            "coverage_ratio": self.coverage_ratio,
            "index_stored": self.index_stored,
            "final_score": self._compute_final_score(),
        })
        return report

    def evaluate_run(self) -> dict:
        epsilon = 1e-6
        score = min(max(self._compute_final_score(), epsilon), 1.0 - epsilon)
        gates = {
            "model_selected": self.selected_model is not None,
            "embedding_executed": self.embedding_done,
            "index_stored": self.index_stored,
            "no_null_embeddings": self.null_embedding_count == 0,
            "submitted": self.submitted,
        }
        return {"approved": all(gates.values()) and score >= 0.6, "gates": gates, "composite_score": score}

    def _dispatch(self, operation: str, parameters: dict) -> tuple[float, dict]:
        if operation == "select_model":
            return self._select_model(parameters)
        elif operation == "configure_batch_size":
            return self._configure_batch_size(parameters)
        elif operation == "run_embedding":
            return self._run_embedding()
        elif operation == "validate_coverage":
            return self._validate_coverage()
        elif operation == "inspect_vectors":
            return self._inspect_vectors()
        elif operation == "store_index":
            return self._store_index()
        elif operation == "submit":
            return self._submit()
        return 0.0, {"error": "unknown_operation"}

    def _select_model(self, params: dict) -> tuple[float, dict]:
        model = params.get("model_name", "")
        if model not in VALID_MODELS:
            return 0.1, {"error": f"invalid_model:{model}", "action": "select_model"}
        self.selected_model = model
        return 0.4, {"error": None, "action": "select_model", "model": model}

    def _configure_batch_size(self, params: dict) -> tuple[float, dict]:
        try:
            bs = int(params.get("batch_size", 32))
            self.batch_size = max(1, min(bs, 256))
            return 0.2, {"error": None, "action": "configure_batch_size", "batch_size": self.batch_size}
        except (ValueError, TypeError):
            return 0.0, {"error": "invalid_batch_size", "action": "configure_batch_size"}

    def _run_embedding(self) -> tuple[float, dict]:
        if not self.selected_model:
            self._record_governance_event("no_model", "high", "run_embedding called without model selected")
            return 0.05, {"error": "model_not_selected", "action": "run_embedding"}
        self.index = {}
        self.null_embedding_count = 0
        for chunk in self.input_chunks:
            cid = chunk.get("chunk_id", f"chunk_{len(self.index):03d}")
            text = chunk.get("text", "")
            if not text.strip():
                self.null_embedding_count += 1
                continue
            self.index[cid] = mock_embed(text)
        self.chunks_embedded = len(self.index)
        total = len(self.input_chunks)
        self.coverage_ratio = self.chunks_embedded / max(total, 1)
        self.embedding_done = True
        reward = 0.4 * self.coverage_ratio + 0.3 * (1.0 if self.null_embedding_count == 0 else 0.3) + 0.2 * (self.step_budget_remaining / max(self.step_budget, 1))
        return reward, {"error": None, "action": "run_embedding", "chunks_embedded": self.chunks_embedded}

    def _validate_coverage(self) -> tuple[float, dict]:
        if not self.embedding_done:
            return 0.1, {"error": "embedding_not_done", "action": "validate_coverage"}
        report = {"coverage_ratio": round(self.coverage_ratio, 4), "null_count": self.null_embedding_count, "total_chunks": len(self.input_chunks)}
        reward = 0.5 if self.coverage_ratio >= 0.95 else 0.2
        return reward, {"error": None, "action": "validate_coverage", "report": report}

    def _inspect_vectors(self) -> tuple[float, dict]:
        sample = {k: v[:5] for k, v in list(self.index.items())[:3]}
        return 0.15, {"error": None, "action": "inspect_vectors", "sample": sample}

    def _store_index(self) -> tuple[float, dict]:
        if not self.embedding_done:
            return 0.05, {"error": "embedding_not_done", "action": "store_index"}
        self.index_stored = True
        return 0.4, {"error": None, "action": "store_index", "index_size": len(self.index)}

    def _submit(self) -> tuple[float, dict]:
        if not self.index_stored:
            self._record_governance_event("premature_submit", "high", "submit called before store_index")
            return 0.05, {"error": "index_not_stored", "action": "submit"}
        self.submitted = True
        self.is_done = True
        final_score = self._compute_final_score()
        return min(final_score + 0.15, 0.99), {"error": None, "action": "submit", "final_score": final_score}

    def _compute_final_score(self) -> float:
        if not self.embedding_done:
            return 0.001
        score = (
            0.4 * self.coverage_ratio +
            0.3 * (1.0 if self.null_embedding_count == 0 else 0.0) +
            0.2 * (1.0 if self.index_stored else 0.0) +
            0.1 * (self.step_budget_remaining / max(self.step_budget, 1))
        )
        return self._clip_reward(score)

    def _load_chunks(self, size: int) -> list[dict]:
        path = Path("data/nexacrm_corpus.json")
        if path.exists():
            with open(path) as f:
                return json.load(f)[:size]
        return [{"chunk_id": f"chunk_{i:03d}", "text": f"NexaCRM entry {i}"} for i in range(size)]
