"""
Tests for Worker Agents 2-5.
All tests must pass before Part 3 begins.
"""

import pytest
from workers.chunking_env import ChunkingEnv
from workers.embedding_env import EmbeddingEnv, mock_embed
from workers.retrieval_env import RetrievalEnv
from workers.evaluation_env import EvaluationEnv


# ------------------------------------------------------------------ #
# Shared contract test helper                                          #
# ------------------------------------------------------------------ #

def assert_worker_contract(worker, task_id: str):
    """Assert full OpenEnv contract for any worker."""
    obs = worker.reset(task_id)
    assert isinstance(obs, dict), "reset() must return dict"
    assert "worker_id" in obs
    assert "step_count" in obs
    assert obs["step_count"] == 0

    first_action = {"operation": worker.VALID_ACTIONS[0], "parameters": {}}
    result = worker.step(first_action)
    assert isinstance(result, tuple) and len(result) == 4
    obs2, reward, done, info = result
    assert 0.0 <= reward <= 1.0, f"Reward out of range: {reward}"
    assert isinstance(done, bool)
    assert isinstance(info, dict)

    s = worker.state()
    assert isinstance(s, dict)
    assert "worker_id" in s

    report = worker.generate_run_report()
    assert isinstance(report, dict)
    assert "action_history" in report

    eval_result = worker.evaluate_run()
    assert "approved" in eval_result
    assert "composite_score" in eval_result
    assert 0.0 <= eval_result["composite_score"] <= 1.0


# ------------------------------------------------------------------ #
# mock_embed Tests                                                     #
# ------------------------------------------------------------------ #

class TestMockEmbed:
    def test_deterministic(self):
        v1 = mock_embed("test text")
        v2 = mock_embed("test text")
        assert v1 == v2

    def test_different_texts_different_vectors(self):
        v1 = mock_embed("hello world")
        v2 = mock_embed("goodbye world")
        assert v1 != v2

    def test_correct_dimension(self):
        v = mock_embed("any text here")
        assert len(v) == 384

    def test_normalized(self):
        import math
        v = mock_embed("normalized vector test")
        norm = math.sqrt(sum(x * x for x in v))
        assert abs(norm - 1.0) < 1e-4


# ------------------------------------------------------------------ #
# ChunkingEnv Tests                                                    #
# ------------------------------------------------------------------ #

class TestChunkingEnv:
    def test_contract(self):
        assert_worker_contract(ChunkingEnv(), "easy_chunking")

    def test_reset_easy(self):
        w = ChunkingEnv()
        obs = w.reset("easy_chunking")
        assert obs["step_budget_remaining"] == 8
        assert obs["strategy"] is None

    def test_reset_hard(self):
        w = ChunkingEnv()
        obs = w.reset("hard_chunking")
        assert obs["step_budget_remaining"] == 12

    def test_set_strategy_valid(self):
        w = ChunkingEnv()
        w.reset("easy_chunking")
        _, reward, _, info = w.step({"operation": "set_strategy", "parameters": {"strategy": "sentence"}})
        assert reward > 0
        assert info["error"] is None
        assert w.strategy == "sentence"

    def test_set_strategy_invalid(self):
        w = ChunkingEnv()
        w.reset("easy_chunking")
        _, reward, _, info = w.step({"operation": "set_strategy", "parameters": {"strategy": "invalid"}})
        assert info["error"] is not None

    def test_set_chunk_size_in_range(self):
        w = ChunkingEnv()
        w.reset("easy_chunking")
        _, reward, _, _ = w.step({"operation": "set_chunk_size", "parameters": {"size": 400}})
        assert reward >= 0.4

    def test_run_chunking_without_strategy(self):
        w = ChunkingEnv()
        w.reset("easy_chunking")
        _, reward, _, info = w.step({"operation": "run_chunking", "parameters": {}})
        assert info["error"] == "strategy_not_set"

    def test_full_episode(self):
        w = ChunkingEnv()
        w.reset("easy_chunking")
        w.step({"operation": "set_strategy", "parameters": {"strategy": "sentence"}})
        w.step({"operation": "set_chunk_size", "parameters": {"size": 400}})
        w.step({"operation": "set_overlap", "parameters": {"overlap": 40}})
        w.step({"operation": "run_chunking", "parameters": {}})
        w.step({"operation": "validate_chunks", "parameters": {}})
        _, reward, done, info = w.step({"operation": "submit", "parameters": {}})
        assert done is True
        assert reward > 0
        assert info["error"] is None

    def test_reward_always_bounded(self):
        w = ChunkingEnv()
        w.reset("medium_chunking")
        for action in w.VALID_ACTIONS[:-1]:
            _, reward, done, _ = w.step({"operation": action, "parameters": {"strategy": "fixed", "size": 300, "overlap": 30}})
            assert 0.0 <= reward <= 1.0
            if done:
                break

    def test_determinism(self):
        w1, w2 = ChunkingEnv(), ChunkingEnv()
        obs1 = w1.reset("easy_chunking")
        obs2 = w2.reset("easy_chunking")
        assert obs1 == obs2

    def test_budget_exhaustion(self):
        w = ChunkingEnv()
        w.reset("easy_chunking")
        done = False
        for _ in range(20):
            _, _, done, _ = w.step({"operation": "preview_chunks", "parameters": {}})
            if done:
                break
        assert done is True


# ------------------------------------------------------------------ #
# EmbeddingEnv Tests                                                   #
# ------------------------------------------------------------------ #

class TestEmbeddingEnv:
    def test_contract(self):
        assert_worker_contract(EmbeddingEnv(), "easy_embedding")

    def test_reset(self):
        w = EmbeddingEnv()
        obs = w.reset("easy_embedding")
        assert obs["selected_model"] is None
        assert obs["chunks_embedded"] == 0

    def test_select_valid_model(self):
        w = EmbeddingEnv()
        w.reset("easy_embedding")
        _, reward, _, info = w.step({"operation": "select_model", "parameters": {"model_name": "all-MiniLM-L6-v2"}})
        assert reward > 0
        assert info["error"] is None

    def test_select_invalid_model(self):
        w = EmbeddingEnv()
        w.reset("easy_embedding")
        _, reward, _, info = w.step({"operation": "select_model", "parameters": {"model_name": "invalid-model"}})
        assert info["error"] is not None

    def test_run_embedding_without_model(self):
        w = EmbeddingEnv()
        w.reset("easy_embedding")
        _, _, _, info = w.step({"operation": "run_embedding", "parameters": {}})
        assert info["error"] == "model_not_selected"

    def test_full_episode(self):
        w = EmbeddingEnv()
        w.reset("easy_embedding")
        w.step({"operation": "select_model", "parameters": {"model_name": "all-MiniLM-L6-v2"}})
        w.step({"operation": "configure_batch_size", "parameters": {"batch_size": 32}})
        w.step({"operation": "run_embedding", "parameters": {}})
        w.step({"operation": "validate_coverage", "parameters": {}})
        w.step({"operation": "store_index", "parameters": {}})
        _, reward, done, info = w.step({"operation": "submit", "parameters": {}})
        assert done is True
        assert reward > 0

    def test_index_populated_after_embedding(self):
        w = EmbeddingEnv()
        w.reset("easy_embedding")
        w.step({"operation": "select_model", "parameters": {"model_name": "all-MiniLM-L6-v2"}})
        w.step({"operation": "run_embedding", "parameters": {}})
        assert len(w.index) > 0

    def test_coverage_ratio_after_embedding(self):
        w = EmbeddingEnv()
        w.reset("easy_embedding")
        w.step({"operation": "select_model", "parameters": {"model_name": "all-MiniLM-L6-v2"}})
        w.step({"operation": "run_embedding", "parameters": {}})
        assert w.coverage_ratio > 0.0

    def test_reward_always_bounded(self):
        w = EmbeddingEnv()
        w.reset("easy_embedding")
        for action in ["select_model", "configure_batch_size", "run_embedding"]:
            params = {"model_name": "all-MiniLM-L6-v2", "batch_size": 32}
            _, reward, done, _ = w.step({"operation": action, "parameters": params})
            assert 0.0 <= reward <= 1.0
            if done:
                break


# ------------------------------------------------------------------ #
# RetrievalEnv Tests                                                   #
# ------------------------------------------------------------------ #

class TestRetrievalEnv:
    def test_contract(self):
        assert_worker_contract(RetrievalEnv(), "easy_retrieval")

    def test_reset(self):
        w = RetrievalEnv()
        obs = w.reset("easy_retrieval")
        assert obs["top_k"] == 5
        assert obs["precision_at_3"] == 0.0

    def test_configure_retrieval_valid(self):
        w = RetrievalEnv()
        w.reset("easy_retrieval")
        _, reward, _, info = w.step({"operation": "configure_retrieval", "parameters": {"top_k": 5}})
        assert reward > 0
        assert info["error"] is None

    def test_run_retrieval(self):
        w = RetrievalEnv()
        w.reset("easy_retrieval")
        w.step({"operation": "configure_retrieval", "parameters": {"top_k": 5}})
        _, reward, _, info = w.step({"operation": "run_retrieval", "parameters": {}})
        assert info["error"] is None
        assert w.retrieval_done is True

    def test_precision_at_3_valid_range(self):
        w = RetrievalEnv()
        w.reset("easy_retrieval")
        w.step({"operation": "configure_retrieval", "parameters": {"top_k": 5}})
        w.step({"operation": "run_retrieval", "parameters": {}})
        assert 0.0 <= w.precision_at_3 <= 1.0

    def test_full_episode(self):
        w = RetrievalEnv()
        w.reset("easy_retrieval")
        w.step({"operation": "configure_retrieval", "parameters": {"top_k": 5}})
        w.step({"operation": "set_reranker", "parameters": {"strategy": "bm25"}})
        w.step({"operation": "run_retrieval", "parameters": {}})
        w.step({"operation": "evaluate_precision", "parameters": {}})
        _, reward, done, info = w.step({"operation": "submit", "parameters": {}})
        assert done is True
        assert reward > 0

    def test_reward_bounded(self):
        w = RetrievalEnv()
        w.reset("easy_retrieval")
        for action in ["configure_retrieval", "set_reranker", "run_retrieval"]:
            params = {"top_k": 5, "strategy": "bm25"}
            _, reward, done, _ = w.step({"operation": action, "parameters": params})
            assert 0.0 <= reward <= 1.0
            if done:
                break


# ------------------------------------------------------------------ #
# EvaluationEnv Tests                                                  #
# ------------------------------------------------------------------ #

class TestEvaluationEnv:
    def test_contract(self):
        assert_worker_contract(EvaluationEnv(), "easy_evaluation")

    def test_reset(self):
        w = EvaluationEnv()
        obs = w.reset("easy_evaluation")
        assert obs["faithfulness_score"] == 0.0
        assert obs["composite_score"] == 0.0

    def test_faithfulness_check(self):
        w = EvaluationEnv()
        w.reset("easy_evaluation")
        _, reward, _, info = w.step({"operation": "run_faithfulness_check", "parameters": {}})
        assert info["error"] is None
        assert w.faithfulness_done is True
        assert 0.0 <= w.faithfulness_score <= 1.0

    def test_pipeline_integrity_check(self):
        w = EvaluationEnv()
        w.reset("easy_evaluation")
        _, reward, _, info = w.step({"operation": "check_pipeline_integrity", "parameters": {}})
        assert info["error"] is None
        assert w.integrity_done is True

    def test_composite_score_bounded(self):
        w = EvaluationEnv()
        w.reset("easy_evaluation")
        w.step({"operation": "run_faithfulness_check", "parameters": {}})
        w.step({"operation": "run_relevance_check", "parameters": {}})
        w.step({"operation": "check_pipeline_integrity", "parameters": {}})
        w.step({"operation": "compute_composite_score", "parameters": {}})
        assert 0.0 < w.composite_score < 1.0

    def test_full_episode(self):
        w = EvaluationEnv()
        w.reset("easy_evaluation")
        w.step({"operation": "run_faithfulness_check", "parameters": {}})
        w.step({"operation": "run_relevance_check", "parameters": {}})
        w.step({"operation": "check_pipeline_integrity", "parameters": {}})
        w.step({"operation": "compute_composite_score", "parameters": {}})
        w.step({"operation": "generate_eval_report", "parameters": {}})
        _, reward, done, info = w.step({"operation": "submit", "parameters": {}})
        assert done is True
        assert reward > 0
        assert info["error"] is None

    def test_submit_before_checks_fails(self):
        w = EvaluationEnv()
        w.reset("easy_evaluation")
        _, _, _, info = w.step({"operation": "submit", "parameters": {}})
        assert info["error"] == "evaluation_incomplete"

    def test_reward_bounded(self):
        w = EvaluationEnv()
        w.reset("easy_evaluation")
        for action in w.VALID_ACTIONS[:-1]:
            _, reward, done, _ = w.step({"operation": action, "parameters": {}})
            assert 0.0 <= reward <= 1.0
            if done:
                break
