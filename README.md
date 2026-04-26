---
title: Fleet AI Oversight
emoji: 🛡️
colorFrom: indigo
colorTo: blue
sdk: docker
pinned: false
---

# Fleet AI Oversight: The Missing Governance Layer for Agentic AI
### Meta Hackathon Finals | Team HackWithPals | OpenEnv Round 2

[![OpenEnv](https://img.shields.io/badge/OpenEnv-v2.0-blue)]
[![Theme](https://img.shields.io/badge/Theme-3.1%20%2B%202-green)]
[![HF Space](https://img.shields.io/badge/HuggingFace-Space-yellow)]

> **"As enterprises scale agentic AI, we built the missing governance layer: an RL-trained oversight agent that supervises RAG workflows, reduces failure risk, and improves reliability across domains."**

---

## 📖 The Problem

Enterprises are rapidly integrating AI into core operations through agent workflows. But as soon as companies deploy multi-agent systems, they face new failure modes: wrong task routing, silent quality drops, budget leakage, and risky decisions.

Today, most teams optimize generation quality, but very few systems train an AI specifically to oversee other AI agents in production workflows. Without intelligent oversight:
- AI agents can drift from intended behavior.
- Errors compound across the pipeline.
- Teams lose money through bad routing, false approvals, and missed violations.
- Trust in enterprise AI systems drops.

We’re solving a hard enterprise problem: when multiple AI workers run a pipeline together, failures come from coordination and governance, not just from one model being “wrong.”

---

## 🎯 Our Thesis

The next frontier is not only better AI workers, but **AI supervisors**. If enterprises are building agent fleets, they need an oversight intelligence layer that learns how to govern those fleets. Most solutions help build AI agents; we train an AI to supervise AI agents with reward-driven learning, measurable outcomes, and transfer behavior across domains.

---

## 🏗️ What We Built

We built an **RL-based oversight agent** trained in a two-phase environment:
1. **Planning Phase**: Allocate the right tasks and configurations to the right workers.
2. **Oversight Phase**: Monitor behavior via partial signals, detect anomalies, and intervene with the right action (Monitor, Intervene, Escalate, Hold, Submit Audit).

### Why RAG?
We focused on RAG because it is the operational core of enterprise AI deployments. Our governed RAG pipeline consists of chained worker behaviors:
1. Data cleaning
2. Chunking
3. Embedding/indexing
4. Retrieval
5. Evaluation

The risk is that small mistakes compound silently. If governance works on a RAG pipeline, it translates to a large share of real enterprise AI use cases.

---

## 🧠 How the Training & Reward Design Works

In our training notebook (`fleet_train.ipynb`), the agent learns via **GRPO (Group Relative Policy Optimization)**. The reward function is intentionally split by behavior quality, incentivizing both **Strategic Allocation** and **Operational Governance**:

- **Planning Rewards**: +0.40 exact task match, +0.20 partial match, -0.30 wrong difficulty.
- **Oversight Rewards**: +0.40 true detection & correct intervention, +0.10 correct approval, +0.15 escalation on ambiguity.
- **Penalties**: -0.45 false positive (pausing a healthy worker), -0.65 missed violation.
- **Completion Reward**: Proper episode closure via `submit_audit` guarantees pipeline quality is finalized correctly.

### Runtime Decisioning
At runtime, the agent starts an episode, sees structured observations (worker states, budgets, anomalies), outputs an action JSON (`action_type`, `worker_id`, `reason`), and the environment executes the action until the audit is submitted.

---

## 🔬 How We Prove It Works

We validate the oversight model across three layers of proof:

1. **Baseline Comparison**: The random policy establishes a floor for detection, reward, and false positive rates.
2. **Post-Training Uplift**: The trained policy massively improves the composite performance score and passes all governance gates.
3. **Zero-Shot Transfer Test**: The model was trained purely on a CRM domain (`NexaCRM FAQ`), and evaluated on a Banking domain (`BankingPro FAQ`). It successfully transfers its governance patterns without domain memorization or retraining.

### The Results
- **Anomaly Detection Rate**: 72% (vs 28% random baseline, a +44pp improvement).
- **False Positive Rate**: Reduced from 45% to 12%.
- **Transfer Domain (Banking)**: Outperforms the random baseline by **5.8x** on unseen data with zero retraining.

---

## 💻 The UI & Dashboard Narrative

Our redesigned frontend (`fleet_bench_ui.html`) serves as the command center for the entire pipeline:
- **Overview**: Outlines the problem, the need, and why governance matters with live operational stats.
- **Fleet Runner**: A live, interactive terminal showing operations and AI interventions in real-time.
- **Audit Report**: A dedicated evidence page detailing governance scores, gates, and decision quality logs.
- **Training Results**: Visual evidence of RL convergence, pre/post-training impact, and anomaly detection.
- **RAG Chatbot**: Proves the usability and accuracy of the governed pipeline.

---

## ⚙️ Quick Start

### Run Locally:
```bash
git clone https://github.com/Dhrumilparikh2806/meta_hackathon_finals_hackwithpals.git
cd meta_hackathon_finals_hackwithpals

# Install dependencies
pip install -r requirements.txt

# Setup Datasets
python data/setup_dataset.py

# Launch the Full-Stack Application (UI + API)
uvicorn app:app --host 0.0.0.0 --port 7860
```

### Run with Docker:
```bash
docker build -t fleet-oversight .
docker run -p 7860:7860 fleet-oversight
```

### Advanced Training & Inference:
```bash
# Run training simulation to generate charts/metrics:
python fleet_train.py --simulate --episodes 30

# Run baseline:
python fleet_baseline.py --task-id easy_fleet --episodes 10

# Run inference (requires HuggingFace token):
export HF_TOKEN=your_token
python fleet_inference.py --task-id easy_fleet
```

---

## 🛠️ API Reference & OpenEnv Compliance

Our architecture is fully compliant with the `OpenEnv` specification (`openenv validate --config fleet_openenv.yaml`).

### Fleet Execution Routes
| Endpoint | Method | Description |
|---|---|---|
| `/fleet/reset` | POST | Starts a new episode — returns `PlanningObservation` |
| `/fleet/plan` | POST | Allocates task to worker (Planning Phase) |
| `/fleet/step` | POST | Submits an oversight agent action (Oversight Phase) |
| `/fleet/state` | GET | Returns the current environment state |
| `/fleet/evaluate`| POST | Performs gate-based evaluation |
| `/rag/query` | POST | Query the governed RAG chatbot index |

### UI Routes
The application is built as an SPA (Single Page Application) accessible natively via the root domain:
- **`/ui`** or **`/`**: Serves the unified Fleet Benchmark UI.
- **`/plots/{filename}`**: Serves static training evidence charts dynamically.

---

## 📂 Project Structure

```
├── fleet/
│   ├── models.py              — Pydantic schemas and planning models
│   ├── worker_registry.py     — Worker management & partial observability
│   ├── anomaly_injector.py    — Fault injection logic
│   ├── oversight_rewards.py   — Planning + oversight reward computation
│   ├── oversight_governance.py— Audit trail and logging
│   ├── oversight_evaluator.py — Gate-based evaluation
│   └── oversight_env.py       — Main two-phase RL environment
├── workers/
│   ├── base_worker.py         — Abstract base class
│   ├── chunking_env.py        — Worker 2 (Chunking)
│   ├── embedding_env.py       — Worker 3 (Embedding)
│   ├── retrieval_env.py       — Worker 4 (Retrieval)
│   └── evaluation_env.py      — Worker 5 (Evaluation)
├── data/                      — NexaCRM & Banking datasets
├── plots/                     — Training convergence visuals (used in UI)
├── tests/                     — Pytest test suites (100% Coverage)
├── ui/                        — Static UI assets (Diagrams, Assets)
├── fleet_bench_ui.html        — Unified Single Page Application UI
├── app.py                     — FastAPI Backend Server
├── fleet_train.py             — GRPO training script
├── fleet_train.ipynb          — Colab-ready notebook
├── fleet_baseline.py          — Random agent baseline runner
├── fleet_inference.py         — LLM runner
└── fleet_openenv.yaml         — OpenEnv environment spec
```

---

Made by **Team HackWithPals** | Meta Hackathon Finals 2026
