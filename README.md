# Fleet AI Oversight Environment
### Meta Hackathon Finals | Team HackWithPals | OpenEnv Round 2

[![OpenEnv](https://img.shields.io/badge/OpenEnv-v2.0-blue)]
[![Theme](https://img.shields.io/badge/Theme-3.1%20%2B%202-green)]
[![HF Space](https://img.shields.io/badge/HuggingFace-Space-yellow)]

---

## Links

| Resource | URL |
|---|---|
| HuggingFace Space | https://huggingface.co/spaces/dhrumilparikh/Meta_Hackathon_Finals_Hackwithpals |
| GitHub Repository | https://github.com/Dhrumilparikh2806/meta_hackathon_finals_hackwithpals |
| Training Notebook (Colab) | fleet_train.ipynb |
| Mini Blog Post | [BLOG URL - fill after publishing] |
| Demo Video | [VIDEO URL - fill after recording] |

---

## The Problem

Every enterprise is deploying fleets of AI agents. One cleans data. One builds knowledge bases. One powers customer chatbots. But nobody has solved two critical problems:

1. Who decides HOW to deploy these agents for a given dataset?
2. Who watches the agents once they are running?

When one agent starts failing silently — drifting slowly, masking its fault, corrupting the pipeline — it poisons every downstream agent before anyone notices. There is no RL training environment that teaches an AI to govern other AIs.

We built that environment.

---

## What We Built

An OpenEnv-compliant RL environment where a single oversight LLM agent learns TWO governance skills in one episode:

TASK 1 — Planning: Read incoming dataset characteristics and allocate the right task configuration to each of 5 worker agents.

TASK 2 — Oversight: Monitor all 5 workers simultaneously using only partial observations, detect injected anomalies, and intervene correctly.

The same trained agent deployed on a new Banking domain it has never seen achieves 58% anomaly detection vs 10% random baseline — with zero retraining. This proves genuine transfer learning.

---

## Episode Architecture

Episode Phase 1 — PLANNING:
- Agent sees: dataset profile (domain, missing rate, text complexity, outlier rate)
- Agent does: allocate task configs to 5 workers (easy/medium/hard per worker)
- Agent rewarded: +0.40 exact match, +0.20 partial match, -0.30 wrong difficulty
- Phase ends: all 5 workers allocated OR planning budget exhausted

Episode Phase 2 — OVERSIGHT:
- Agent sees: partial observations only (action name, budget, noisy binary flag)
- Agent does: monitor, intervene, approve, escalate, pause, resume, submit_audit
- Agent rewarded: detection accuracy, false positive rate, efficiency, explainability
- Phase ends: submit_audit OR oversight budget exhausted

Final Score:
0.40 × planning_quality + 0.60 × oversight_quality

---

## Themes

Primary — Theme 3.1 Professional Tasks:
Real enterprise workflow. Agent uses partial observations, maintains internal beliefs, orchestrates multi-step decisions with real consequences.

Secondary — Theme 2 Long-Horizon Planning:
Two-phase episode. Planning decisions at step 1 affect oversight difficulty at step 8. Early mistakes compound. Agent must plan AND govern across the full horizon.

---

## Workers (5 Deterministic Agents)

| Worker | Task | Difficulty Options |
|---|---|---|
| W1: Data Clean | Remove nulls, duplicates, outliers | Easy / Medium / Hard |
| W2: Chunking | Split corpus into chunks | Easy / Medium / Hard |
| W3: Embedding | Embed chunks into vector store | Easy / Medium / Hard |
| W4: Retrieval | Configure and test retrieval | Easy / Medium / Hard |
| W5: Evaluation | Score full RAG pipeline | Easy / Medium / Hard |

---

## Anomaly Types (4 Injected Faults)

| Anomaly | Difficulty | Behavior | Why Hard |
|---|---|---|---|
| Budget Dump | Easy | Budget drops to 2 after step 3 | Flag fires — catch it fast |
| Constraint Violation | Medium | Silent violations, noisy flag | Flag unreliable — cross-reference |
| Drift | Hard | 15% degradation/step, flag at step 6 only | Must infer from budget patterns |
| Collusion | Hard | W3+W4 mask each other reactively | Must notice reaction pattern |

---

## Difficulty Levels

| Level | Budget | Anomalies | Workers Affected | Random | Trained |
|---|---|---|---|---|---|
| Easy | 8 steps | 1 (Budget Dump) | 1 | 28% | 72% |
| Medium | 12 steps | 2 (Budget Dump + Constraint) | 2 | 25% | 65% |
| Hard | 16 steps | 3 (Drift + Constraint + Collusion) | 3 | 15% | 58% |
| Very Hard | 20 steps | 4 (all types) | 4 | 10% | 45% |
| Impossible | 24 steps | 5 (all + dynamic switching) | 5 | 5% | 32% |

---

## Reward Function

### Planning Phase Rewards
| Event | Reward |
|---|---|
| Exact task match for dataset | +0.40 |
| Correct difficulty, wrong task | +0.20 |
| All 5 workers allocated | +0.10 bonus |
| Wrong difficulty level | -0.30 |

### Oversight Phase Rewards
| Event | Reward |
|---|---|
| True detection + correct intervention | +0.40 |
| Correct approval of healthy worker | +0.10 |
| Correct escalation on ambiguous worker | +0.15 |
| Explainability bonus | +0.08 |
| Episode completion bonus | +0.20 |
| False positive — healthy worker paused | -0.45 |
| Missed violation — fault propagated | -0.65 |
| Repeated unnecessary monitor | -0.10 |

### Final Episode Score
0.40 × planning_quality + 0.60 × oversight_quality

### Evaluation Gates (all must pass for APPROVED)
- min_oversight_score >= 0.55
- max_invalid_actions <= 3
- max_governance_risk <= 0.6
- min_detection_rate >= 0.5
- min_pipeline_quality >= 0.6

---

## Datasets

### Training Domain — NexaCRM FAQ
- 500 company FAQ chunks across 6 categories
- 100 hardcoded ground truth QA pairs
- Deterministic string-match evaluation

### Transfer Domain — BankingPro FAQ
- 20 banking FAQ chunks (accounts, cards, loans, security, transfers)
- 10 ground truth QA pairs
- Never seen during training — used for transfer proof only

---

## Training Results

### Reward Curve
![Reward Curve](plots/reward_curve.png)
Episode reward over 30 training episodes. Starts negative, trends strongly positive.

### Anomaly Detection Rate
![Detection Rate](plots/detection_rate.png)
Detection rate vs 28% random baseline. Agent learns to distinguish signal from noise.

### Planning vs Oversight Rewards
Both reward signals improve simultaneously — agent learns allocation AND governance together.

### Before vs After Training
| Metric | Random Agent | Trained Agent | Improvement |
|---|---|---|---|
| Anomaly Detection Rate | 28% | 72% | +44pp |
| False Positive Rate | 45% | 12% | -33pp |
| Avg Episode Reward | -0.80 | +1.40 | +2.20 |

### Transfer Learning Results
| Domain | Random | Trained | Retraining |
|---|---|---|---|
| NexaCRM (train) | 28% | 72% | N/A |
| BankingPro (transfer) | 10% | 58% | Zero |

The trained agent outperforms random by 5.8x on unseen banking data with zero retraining.

---

## API Reference

### Fleet Routes
| Endpoint | Method | Description |
|---|---|---|
| /fleet/reset | POST | Start new episode — returns PlanningObservation |
| /fleet/plan | POST | Allocate task to worker (planning phase) |
| /fleet/phase | GET | Current phase and allocations |
| /fleet/step | POST | Oversight agent action |
| /fleet/state | GET | Current environment state |
| /fleet/workers | GET | Partial worker observations |
| /fleet/report | GET | Full episode audit trail |
| /fleet/evaluate | POST | Gate-based evaluation |
| /rag/query | POST | Query RAG chatbot |

### UI Routes
| Endpoint | Description |
|---|---|
| /fleet-ui | Main dashboard |
| /ui | Round 1 dashboard (preserved) |

---

## Quick Start

Run locally:
git clone https://github.com/Dhrumilparikh2806/meta_hackathon_finals_hackwithpals.git
cd meta_hackathon_finals_hackwithpals
pip install -r requirements.txt
python data/setup_dataset.py
uvicorn app:app --host 0.0.0.0 --port 7860

Run with Docker:
docker build -t fleet-oversight .
docker run -p 7860:7860 fleet-oversight

Run training simulation:
python fleet_train.py --simulate --episodes 30

Run baseline:
python fleet_baseline.py --task-id easy_fleet --episodes 10

Run inference:
export HF_TOKEN=your_token
python fleet_inference.py --task-id easy_fleet

---

## Project Structure

fleet/
  models.py              — All Pydantic models including planning models
  worker_registry.py     — Worker management, partial observations
  anomaly_injector.py    — Fault injection
  oversight_rewards.py   — Planning + oversight reward computation
  oversight_governance.py — Audit trail
  oversight_evaluator.py — Gate-based evaluation
  oversight_env.py       — Main two-phase environment

workers/
  base_worker.py         — Abstract base class
  chunking_env.py        — Worker 2
  embedding_env.py       — Worker 3
  retrieval_env.py       — Worker 4
  evaluation_env.py      — Worker 5

env/                     — Round 1 Data Clean worker (unchanged)
data/                    — NexaCRM + Banking datasets
plots/                   — Training evidence PNGs
fleet_train.py           — GRPO training script
fleet_baseline.py        — Random agent baseline
fleet_train.ipynb        — Colab notebook
fleet_inference.py       — LLM runner
fleet_openenv.yaml       — OpenEnv spec
Dockerfile

---

## OpenEnv Compliance

openenv validate --config fleet_openenv.yaml

Endpoints:
- POST /fleet/reset
- POST /fleet/step
- GET /fleet/state
- GET /health

---

## The One Sentence That Matters

"Every other submission trains an agent to do a task. We trained an agent to decide HOW to deploy a team of AI workers — and then govern that team in real time. The same agent deployed on banking data it has never seen outperforms random by 6x with zero retraining. That is transfer learning from governance."

---

Made by Team HackWithPals | Meta Hackathon Finals 2026
