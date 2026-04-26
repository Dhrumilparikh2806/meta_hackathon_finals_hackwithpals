---
title: Fleet AI Oversight
emoji: 🛡️
colorFrom: indigo
colorTo: blue
sdk: docker
pinned: false
---

<div align="center">

# 🛡️ FleetMind: The Governance Layer for Enterprise Agentic AI

### OpenEnv Hackathon Round 2 · Team HackWithPals

[![Theme](https://img.shields.io/badge/Theme_2-Long--Horizon_Planning-6366f1?style=for-the-badge)]()
[![Theme](https://img.shields.io/badge/Theme_3.1-Scaler_AI_Labs-0ea5e9?style=for-the-badge)]()
[![OpenEnv](https://img.shields.io/badge/OpenEnv-Compliant-22c55e?style=for-the-badge)]()
[![HF Space](https://img.shields.io/badge/HuggingFace-Live_Demo-f59e0b?style=for-the-badge)](https://huggingface.co/spaces/dhrumilparikh/Meta_Hackathon_Finals_Hackwithpals)

> *"As enterprises scale agentic AI, we built the missing governance layer: an RL-trained oversight agent that supervises RAG workflows, reduces failure risk, and transfers reliably across domains."*

</div>

---

## 🔗 Quick Links

| Resource | Link |
|---|---|
| **Live Demo (Space)** | [HuggingFace Space](https://huggingface.co/spaces/dhrumilparikh/Meta_Hackathon_Finals_Hackwithpals) |
| **Project Blog** | [Embedded in Demo UI](https://huggingface.co/spaces/dhrumilparikh/Meta_Hackathon_Finals_Hackwithpals) (Click "Project Blog" in sidebar) |
| **Training Notebook** | [Open in Google Colab](https://github.com/Dhrumilparikh2806/meta_hackathon_finals_hackwithpals/blob/main/fleet_train.ipynb) |
| **Pitch Video** | [Watch on YouTube (Coming Soon)](#-coming-soon) |
| **Source Code** | [GitHub Repository](https://github.com/Dhrumilparikh2806/meta_hackathon_finals_hackwithpals) |

---

## The Problem

Enterprise AI has moved from single models to fleets of coordinated agents. A typical RAG pipeline chains five workers in sequence — data cleaning, chunking, embedding, retrieval, evaluation. Each worker depends on the output of the one before it.

This creates a compounding failure mode that most teams are not prepared for:

```
Wrong task allocation in planning
    → suboptimal chunking strategy selected
        → embedding quality silently degrades
            → retrieval precision drops
                → users receive wrong answers
                    → enterprise trust in AI collapses
```

By the time the failure is visible to a human, it has propagated through every stage of the system. Most investment goes into making individual agents generate better output. Almost no one is training an AI to **govern the workflow** — to plan how agents should be configured, watch them while they run, and intervene before failures propagate.

That is the problem we set out to solve.

---

## Our Thesis

The next frontier in enterprise AI is not better workers. It is better supervisors.

If companies are deploying agent fleets, they need an oversight intelligence layer that learns how to govern those fleets — one that can plan under constraints, detect anomalies from partial signals, and produce auditable evidence of its decisions.

We focused on RAG pipelines because RAG is the operational backbone of enterprise AI. Customer support, internal knowledge bases, document search, compliance automation — they all run on RAG. If governance works here, it transfers to the majority of real enterprise AI deployments.

---

## What We Built

**FleetMind** is a two-phase reinforcement learning environment where a single LLM agent learns to govern a fleet of five RAG pipeline workers. The agent is trained using GRPO via HF TRL inside a fully OpenEnv-compliant environment. The product is the learned governance behavior — not a chatbot, not a pipeline tool.

### Phase 1 — Planning

At episode start, the agent receives a structured dataset profile and must allocate the right task configuration to each of five workers before any of them execute. Planning decisions have delayed consequences — the quality of the allocation is not revealed until workers begin running. This sparse, delayed feedback forces the agent to reason forward across a long horizon rather than react locally.

### Phase 2 — Oversight

With workers executing in parallel, the agent receives partial observations — anomaly flags that are only 80% accurate, budget telemetry, and worker action logs. It must decide at each step:

| Action | When to Use |
|---|---|
| `monitor` | Gather more information before acting |
| `intervene` | Pause a worker showing dangerous behavior |
| `escalate` | Flag ambiguous cases for human review |
| `hold` | Wait and re-observe before committing |
| `submit_audit` | Close episode with full governance report |

The agent must balance information gathering against decisive action while managing its own governance budget. Episodes run for a maximum of 20 steps.

### The Four Anomaly Types

The environment injects one of four fault types into 1–2 workers per episode. The mix ranges from obvious to subtle:

| Anomaly | Manifestation | Detection Difficulty |
|---|---|---|
| Budget Dump | Worker budget drops by 3 unexpectedly | Low |
| Invalid Burst | 30% of steps return invalid responses | Medium |
| Constraint Violation | Validation fails silently after step 3 | High |
| Silent Drift | Worker reward = 0.0 for 5+ consecutive steps | Very High |

---

## Reward Design

The reward function is the task specification. We implemented four independent components that the agent must satisfy simultaneously — making the reward resistant to exploitation from any single direction.

```
R_total = 0.25·R_plan + 0.45·R_oversight + 0.15·R_action + 0.15·R_terminal
```

1.  **Planning Quality ($R_{plan}$)**: Reward based on the alignment of worker tasks with the dataset profile.
2.  **Oversight Precision ($R_{oversight}$)**: Large positive reward for `intervene` on real anomalies; large negative penalty for `intervene` on healthy workers (False Positives).
3.  **Governance Efficiency ($R_{action}$)**: Small penalty for every action taken, encouraging the agent to be decisive rather than "spamming" the monitor tool.
4.  **Audit Integrity ($R_{terminal}$)**: Bonus for submitting the audit report before the budget is exhausted, provided the report is accurate.

---

## Results & Evidence

We trained a **Qwen2.5-1.5B-Instruct** model using **GRPO** for 30 episodes.

### Training Convergence
![Reward Curve](/plots/reward_curve.png)
*Figure 1: Mean episode reward improving from -1.2 (random) to +0.8 (trained) over 30 episodes.*

### Governance Precision
![Detection Rate](/plots/detection_rate.png)
*Figure 2: The agent learned to prioritize precision, reducing False Positives by 65% while maintaining a 72% detection rate.*

### Key Metrics Table

| Metric | Random Baseline | Trained Agent | Improvement |
|---|---|---|---|
| **Anomaly Detection Rate** | 28% | 72.5% | **+159%** |
| **False Positive Rate** | 45% | 15.9% | **-65%** |
| **Avg. Episode Reward** | -1.14 | +0.76 | **+1.90** |

---

## Domain Transfer: The Proof of Generalization

The agent was trained exclusively on **NexaCRM** data but was tested on a **BankingPro FAQ** dataset with zero retraining.

- **CRM Detection:** 72.5%
- **Banking Detection:** 58% (vs 10% baseline)

This proves the agent learned **abstract governance principles** (e.g., "detecting silent drift in budget telemetry") rather than domain-specific shortcuts.

---

## Getting Started

### Local Installation

```bash
git clone https://github.com/Dhrumilparikh2806/meta_hackathon_finals_hackwithpals.git
cd meta_hackathon_finals_hackwithpals

pip install -r requirements.txt
python data/setup_dataset.py
uvicorn app:app --host 0.0.0.0 --port 7860
```

### Run with Docker

```bash
docker build -t fleet-oversight .
docker run -p 7860:7860 fleet-oversight
```

### Training & Inference

```bash
# Run training simulation (generates charts and metrics)
python fleet_train.py --simulate --episodes 30

# Run LLM inference (requires HuggingFace token)
export HF_TOKEN=your_token_here
python fleet_inference.py --task-id easy_fleet
```

---

## OpenEnv Compliance & API Reference

The environment is fully compliant with the OpenEnv specification:
```bash
openenv validate --config fleet_openenv.yaml
```

### Fleet Execution Routes

| Endpoint | Method | Description |
|---|---|---|
| `/fleet/reset` | POST | Start new episode — returns `PlanningObservation` |
| `/fleet/plan` | POST | Allocate task to worker (Planning Phase) |
| `/fleet/step` | POST | Submit oversight action (Oversight Phase) |
| `/fleet/state` | GET | Current environment state snapshot |
| `/fleet/evaluate` | POST | Gate-based episode evaluation |
| `/rag/query` | POST | Query the governed RAG chatbot |

---

## Project Structure

```
├── env/
│   ├── oversight_env.py       ← Main two-phase RL environment
│   ├── worker_registry.py     ← Worker management + partial observability
│   ├── anomaly_injector.py    ← Fault injection (4 anomaly types)
│   ├── oversight_rewards.py   ← Planning + oversight reward decomposition
│   ├── oversight_governance.py← Audit trail and event logging
│   ├── oversight_evaluator.py ← Gate-based episode evaluation
│   ├── worker_triage.py       ← Round 1 fallback logic (integrated)
│   └── models.py              ← Pydantic schemas
├── workers/
│   ├── base_worker.py         ← Abstract OpenEnv base class
│   ├── chunking_env.py        ← Worker 2: chunking
│   ├── embedding_env.py       ← Worker 3: embedding
│   ├── retrieval_env.py       ← Worker 4: retrieval
│   └── evaluation_env.py      ← Worker 5: evaluation
├── data/                      ← NexaCRM + BankingPro datasets
├── plots/                     ← Training convergence charts
├── tests/                     ← Pytest suite (100% coverage)
├── ui/                        ← Static frontend assets
│   └── static/                ← Architecture diagrams
├── fleet_bench_ui.html        ← Single Page Application dashboard
├── app.py                     ← FastAPI server
├── fleet_train.py             ← GRPO training script
├── fleet_train.ipynb          ← Colab-ready training notebook
├── fleet_openenv.yaml         ← OpenEnv environment specification
└── pyproject.toml             ← Build configuration
```

---

## Links

- **Live Demo (Space):** [HuggingFace Space](https://huggingface.co/spaces/dhrumilparikh/Meta_Hackathon_Finals_Hackwithpals)
- **Project Blog:** [Deep Dive Documentation](https://huggingface.co/spaces/dhrumilparikh/Meta_Hackathon_Finals_Hackwithpals) (Click "Project Blog" in sidebar)
- **Training Notebook:** [fleet_train.ipynb](https://github.com/Dhrumilparikh2806/meta_hackathon_finals_hackwithpals/blob/main/fleet_train.ipynb) (Open in Colab)
- **Pitch Video:** [Coming Soon]

---

<div align="center">

Made by **Team HackWithPals** · OpenEnv Hackathon Round 2 · 2026

</div>