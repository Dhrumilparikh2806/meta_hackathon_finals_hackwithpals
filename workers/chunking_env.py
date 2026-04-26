"""
Worker Agent 2 — Chunking Agent
Task: Split cleaned text corpus into optimal chunks for RAG retrieval.
The agent chooses strategy, chunk size, and overlap to maximize retrieval quality.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Optional

from workers.base_worker import BaseWorker

# ------------------------------------------------------------------ #
# Task Configurations                                                  #
# ------------------------------------------------------------------ #

CHUNKING_TASK_CONFIGS = {
    "easy_chunking": {
        "task_id": "easy_chunking",
        "difficulty": "easy",
        "step_budget": 8,
        "target_chunk_size_min": 200,
        "target_chunk_size_max": 600,
        "target_overlap_max": 0.20,
        "corpus_size": 50,
        "description": "Plain text corpus, straightforward chunking",
    },
    "medium_chunking": {
        "task_id": "medium_chunking",
        "difficulty": "medium",
        "step_budget": 10,
        "target_chunk_size_min": 200,
        "target_chunk_size_max": 600,
        "target_overlap_max": 0.15,
        "corpus_size": 100,
        "description": "Mixed content corpus with varying lengths",
    },
    "hard_chunking": {
        "task_id": "hard_chunking",
        "difficulty": "hard",
        "step_budget": 12,
        "target_chunk_size_min": 150,
        "target_chunk_size_max": 500,
        "target_overlap_max": 0.10,
        "corpus_size": 200,
        "description": "Noisy text with inconsistent formatting",
    },
}

VALID_STRATEGIES = ["fixed", "sentence", "paragraph", "semantic"]


class ChunkingEnv(BaseWorker):
    """
    Worker Agent 2: Chunking Agent.
    
    Actions:
    - set_strategy: Choose chunking strategy (fixed/sentence/paragraph/semantic)
    - set_chunk_size: Set target chunk size in tokens
    - set_overlap: Set overlap between chunks (0-100 tokens)
    - preview_chunks: Preview first 3 chunks with current settings
    - run_chunking: Execute chunking on full corpus
    - validate_chunks: Check chunk quality metrics
    - submit: Submit final chunks and receive terminal reward
    """

    VALID_ACTIONS = [
        "set_strategy", "set_chunk_size", "set_overlap",
        "preview_chunks", "run_chunking", "validate_chunks", "submit"
    ]

    def __init__(self) -> None:
        super().__init__(worker_id="worker_2", worker_name="Chunking Agent")

        # Chunking state
        self.strategy: Optional[str] = None
        self.chunk_size: int = 512          # default
        self.overlap: int = 50              # default tokens
        self.chunks: list[dict] = []
        self.chunk_count: int = 0
        self.avg_chunk_size: float = 0.0
        self.overlap_ratio: float = 0.0
        self.empty_chunk_count: int = 0
        self.chunking_done: bool = False
        self.validation_report: dict = {}
        self.task_config: dict = {}
        self.corpus: list[dict] = []

    # ------------------------------------------------------------------ #
    # OpenEnv Contract                                                     #
    # ------------------------------------------------------------------ #

    def reset(self, task_id: str) -> dict:
        self._reset_episode_tracking()
        self.task_id = task_id
        self.task_config = CHUNKING_TASK_CONFIGS.get(
            task_id, CHUNKING_TASK_CONFIGS["easy_chunking"]
        )
        self.step_budget = self.task_config["step_budget"]
        self.step_budget_remaining = self.step_budget

        # Reset chunking state
        self.strategy = None
        self.chunk_size = 512
        self.overlap = 50
        self.chunks = []
        self.chunk_count = 0
        self.avg_chunk_size = 0.0
        self.overlap_ratio = 0.0
        self.empty_chunk_count = 0
        self.chunking_done = False
        self.validation_report = {}

        # Load corpus
        self.corpus = self._load_corpus(self.task_config["corpus_size"])

        return self.state()

    def step(self, action_dict: dict) -> tuple[dict, float, bool, dict]:
        operation = action_dict.get("operation", "")
        parameters = action_dict.get("parameters", {})

        if self.is_done:
            return self.state(), 0.0, True, {"error": "episode_already_done", "action": operation}

        if operation not in self.VALID_ACTIONS:
            self._record_governance_event("invalid_action", "medium", f"Unknown action: {operation}")
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

        # Dispatch action
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
            "strategy": self.strategy,
            "chunk_size": self.chunk_size,
            "overlap": self.overlap,
            "chunk_count": self.chunk_count,
            "avg_chunk_size": round(self.avg_chunk_size, 2),
            "overlap_ratio": round(self.overlap_ratio, 4),
            "empty_chunk_count": self.empty_chunk_count,
            "chunking_done": self.chunking_done,
            "validation_report": self.validation_report,
            "corpus_size": len(self.corpus),
            "task_config": self.task_config,
        })
        return base

    def generate_run_report(self) -> dict:
        report = self._get_base_report()
        report.update({
            "strategy": self.strategy,
            "chunk_size": self.chunk_size,
            "overlap": self.overlap,
            "chunk_count": self.chunk_count,
            "avg_chunk_size": self.avg_chunk_size,
            "empty_chunk_count": self.empty_chunk_count,
            "chunking_done": self.chunking_done,
            "final_score": self._compute_final_score(),
        })
        return report

    def evaluate_run(self) -> dict:
        epsilon = 1e-6
        score = self._compute_final_score()
        score = min(max(score, epsilon), 1.0 - epsilon)
        gates = {
            "strategy_set": self.strategy is not None,
            "chunking_executed": self.chunking_done,
            "no_empty_chunks": self.empty_chunk_count == 0,
            "submitted": self.submitted,
        }
        approved = all(gates.values()) and score >= 0.6
        return {"approved": approved, "gates": gates, "composite_score": score}

    # ------------------------------------------------------------------ #
    # Action Dispatch                                                      #
    # ------------------------------------------------------------------ #

    def _dispatch(self, operation: str, parameters: dict) -> tuple[float, dict]:
        if operation == "set_strategy":
            return self._set_strategy(parameters)
        elif operation == "set_chunk_size":
            return self._set_chunk_size(parameters)
        elif operation == "set_overlap":
            return self._set_overlap(parameters)
        elif operation == "preview_chunks":
            return self._preview_chunks()
        elif operation == "run_chunking":
            return self._run_chunking()
        elif operation == "validate_chunks":
            return self._validate_chunks()
        elif operation == "submit":
            return self._submit()
        return 0.0, {"error": "unknown_operation"}

    def _set_strategy(self, params: dict) -> tuple[float, dict]:
        strategy = params.get("strategy", "")
        if strategy not in VALID_STRATEGIES:
            self._record_governance_event("bad_strategy", "low", f"Invalid strategy: {strategy}")
            return 0.0, {"error": f"invalid_strategy:{strategy}", "action": "set_strategy"}
        self.strategy = strategy
        reward = 0.3 if strategy in ("sentence", "paragraph") else 0.2
        return reward, {"error": None, "action": "set_strategy", "strategy": strategy}

    def _set_chunk_size(self, params: dict) -> tuple[float, dict]:
        size = params.get("size", 512)
        try:
            size = int(size)
        except (ValueError, TypeError):
            return 0.0, {"error": "invalid_chunk_size", "action": "set_chunk_size"}
        self.chunk_size = size
        config = self.task_config
        if config["target_chunk_size_min"] <= size <= config["target_chunk_size_max"]:
            reward = 0.5
        elif size < config["target_chunk_size_min"] * 0.5 or size > config["target_chunk_size_max"] * 2:
            reward = 0.05
            self._record_governance_event("bad_chunk_size", "medium", f"Chunk size {size} far out of range")
        else:
            reward = 0.2
        return reward, {"error": None, "action": "set_chunk_size", "size": size}

    def _set_overlap(self, params: dict) -> tuple[float, dict]:
        overlap = params.get("overlap", 50)
        try:
            overlap = int(overlap)
        except (ValueError, TypeError):
            return 0.0, {"error": "invalid_overlap", "action": "set_overlap"}
        self.overlap = max(0, min(overlap, self.chunk_size // 2))
        overlap_ratio = self.overlap / max(self.chunk_size, 1)
        reward = 0.4 if overlap_ratio <= self.task_config["target_overlap_max"] else 0.1
        return reward, {"error": None, "action": "set_overlap", "overlap": self.overlap}

    def _preview_chunks(self) -> tuple[float, dict]:
        if not self.corpus:
            return 0.1, {"error": None, "action": "preview_chunks", "preview": []}
        sample_text = self.corpus[0]["text"]
        preview = self._chunk_text(sample_text)[:3]
        return 0.2, {"error": None, "action": "preview_chunks", "preview": preview}

    def _run_chunking(self) -> tuple[float, dict]:
        if self.strategy is None:
            self._record_governance_event("no_strategy", "high", "run_chunking called without strategy set")
            return 0.05, {"error": "strategy_not_set", "action": "run_chunking"}

        all_chunks = []
        for doc in self.corpus:
            chunks = self._chunk_text(doc["text"])
            all_chunks.extend([
                {"chunk_id": f"chunk_{len(all_chunks) + i:03d}", "text": c, "source_doc": doc.get("chunk_id", "unknown")}
                for i, c in enumerate(chunks)
            ])

        self.chunks = all_chunks
        self.chunk_count = len(all_chunks)
        self.empty_chunk_count = sum(1 for c in all_chunks if not c["text"].strip())
        sizes = [len(c["text"].split()) for c in all_chunks if c["text"].strip()]
        self.avg_chunk_size = sum(sizes) / max(len(sizes), 1)
        self.overlap_ratio = self.overlap / max(self.chunk_size, 1)
        self.chunking_done = True

        reward = self._compute_chunking_reward()
        return reward, {"error": None, "action": "run_chunking", "chunk_count": self.chunk_count}

    def _validate_chunks(self) -> tuple[float, dict]:
        if not self.chunking_done:
            return 0.1, {"error": "chunking_not_done", "action": "validate_chunks"}
        config = self.task_config
        size_ok = config["target_chunk_size_min"] <= self.avg_chunk_size <= config["target_chunk_size_max"]
        overlap_ok = self.overlap_ratio <= config["target_overlap_max"]
        no_empties = self.empty_chunk_count == 0
        self.validation_report = {
            "size_in_range": size_ok,
            "overlap_acceptable": overlap_ok,
            "no_empty_chunks": no_empties,
            "chunk_count": self.chunk_count,
            "avg_chunk_size": round(self.avg_chunk_size, 2),
        }
        reward = 0.3 if (size_ok and overlap_ok and no_empties) else 0.15
        return reward, {"error": None, "action": "validate_chunks", "report": self.validation_report}

    def _submit(self) -> tuple[float, dict]:
        if not self.chunking_done:
            self._record_governance_event("premature_submit", "high", "submit called before run_chunking")
            return 0.05, {"error": "chunking_not_done", "action": "submit"}
        self.submitted = True
        self.is_done = True
        final_score = self._compute_final_score()
        terminal_reward = min(final_score + 0.15, 0.99)
        return terminal_reward, {"error": None, "action": "submit", "final_score": final_score}

    # ------------------------------------------------------------------ #
    # Internal Helpers                                                     #
    # ------------------------------------------------------------------ #

    def _chunk_text(self, text: str) -> list[str]:
        """Deterministic chunking based on selected strategy."""
        words = text.split()
        if not words:
            return []
        if self.strategy in ("fixed", None):
            return self._fixed_chunk(words)
        elif self.strategy == "sentence":
            return self._sentence_chunk(text)
        elif self.strategy == "paragraph":
            return self._paragraph_chunk(text)
        elif self.strategy == "semantic":
            return self._fixed_chunk(words)  # Mock semantic = fixed for determinism
        return self._fixed_chunk(words)

    def _fixed_chunk(self, words: list[str]) -> list[str]:
        step = max(self.chunk_size - self.overlap, 1)
        chunks = []
        i = 0
        while i < len(words):
            chunk = " ".join(words[i:i + self.chunk_size])
            if chunk.strip():
                chunks.append(chunk)
            i += step
        return chunks

    def _sentence_chunk(self, text: str) -> list[str]:
        sentences = [s.strip() for s in text.replace(".", ". ").split(". ") if s.strip()]
        chunks, current = [], []
        current_len = 0
        for sent in sentences:
            sent_len = len(sent.split())
            if current_len + sent_len > self.chunk_size and current:
                chunks.append(" ".join(current))
                current = current[-2:] if self.overlap > 0 else []
                current_len = sum(len(s.split()) for s in current)
            current.append(sent)
            current_len += sent_len
        if current:
            chunks.append(" ".join(current))
        return chunks if chunks else [text]

    def _paragraph_chunk(self, text: str) -> list[str]:
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        if not paragraphs:
            paragraphs = [text]
        chunks, current = [], []
        current_len = 0
        for para in paragraphs:
            para_len = len(para.split())
            if current_len + para_len > self.chunk_size and current:
                chunks.append("\n\n".join(current))
                current = []
                current_len = 0
            current.append(para)
            current_len += para_len
        if current:
            chunks.append("\n\n".join(current))
        return chunks

    def _compute_chunking_reward(self) -> float:
        config = self.task_config
        size_score = 1.0 if (
            config["target_chunk_size_min"] <= self.avg_chunk_size <= config["target_chunk_size_max"]
        ) else max(0.0, 1.0 - abs(self.avg_chunk_size - 400) / 400)
        overlap_score = 1.0 if self.overlap_ratio <= config["target_overlap_max"] else 0.3
        completeness_score = 1.0 if self.empty_chunk_count == 0 else max(0.0, 1.0 - self.empty_chunk_count / max(self.chunk_count, 1))
        return 0.4 * size_score + 0.3 * overlap_score + 0.2 * completeness_score + 0.1 * (self.step_budget_remaining / max(self.step_budget, 1))

    def _compute_final_score(self) -> float:
        if not self.chunking_done:
            return 0.001
        return self._clip_reward(self._compute_chunking_reward())

    def _load_corpus(self, size: int) -> list[dict]:
        corpus_path = Path("data/nexacrm_corpus.json")
        if corpus_path.exists():
            with open(corpus_path) as f:
                corpus = json.load(f)
            return corpus[:size]
        # Fallback: generate minimal synthetic corpus
        return [
            {"chunk_id": f"chunk_{i:03d}", "text": f"NexaCRM FAQ entry {i}: This covers feature {i} configuration and usage.", "source": "synthetic"}
            for i in range(size)
        ]
