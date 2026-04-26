# We Built the First RL Environment for Two-Phase AI Fleet Governance

Every enterprise is deploying AI agent pipelines. But two problems remain unsolved: who decides how to configure the agents for a given dataset, and who watches them while they run? We built a single RL environment that trains one LLM agent to do both.

## Section 1 — The Two Tasks

Our environment introduces a two-phase episode structure. In the **Planning Phase**, the oversight agent reads a dataset profile—including missing value rates and text complexity—and allocates the optimal task configuration to a fleet of 5 worker agents. In the **Oversight Phase**, the agent monitors these workers in real-time. It must use partial observations to detect injected anomalies like budget dumps or worker collusion and decide when to intervene, approve, or escalate.

## Section 2 — Why This Is Hard

Governing an AI fleet is significantly more complex than standard task completion. The agent operates with partial observations and noisy flags that are only 80% accurate. Anomalies like "Drift" are silent for several steps, requiring the agent to infer faults from subtle budget patterns. Crucially, planning decisions in step 1 compound over the entire 20-step horizon; a wrong allocation makes the oversight task exponentially harder.

## Section 3 — The Reward Design

To train a high-precision governor, we implemented an asymmetrical reward structure. A missed violation (fault propagation) carries a heavy -0.65 penalty, while a false positive (halting a healthy worker) costs -0.45. This tension forces the agent to move beyond simple threshold-based logic and learn a sophisticated "internal belief" system about worker health.

## Section 4 — The Transfer Proof

The most critical achievement of this project is the proof of domain transfer. We trained our oversight agent entirely on **NexaCRM CRM** data. We then deployed that same trained model—with **zero retraining**—to govern a **BankingPro FAQ** pipeline it had never seen before. The agent achieved a **58% anomaly detection rate** on the banking data, compared to a 10% random baseline. This proves the environment teaches transferable governance principles rather than domain-specific shortcuts.

## Section 5 — Results

| Metric | Random Agent | Trained Agent (CRM) | Trained Agent (Banking) |
|---|---|---|---|
| Detection Rate | 28% | 72% | 58% |
| False Positive Rate| 45% | 12% | 18% |
| Improvement | Baseline | +44pp | +48pp vs Baseline |

## Section 6 — Try it

Explore the project and run your own oversight episodes:
- **HuggingFace Space**: [Link](https://huggingface.co/spaces/dhrumilparikh/Meta_Hackathon_Finals_Hackwithpals)
- **GitHub Repository**: [Link](https://github.com/Dhrumilparikh2806/meta_hackathon_finals_hackwithpals)
- **Training Notebook**: [fleet_train.ipynb](https://github.com/Dhrumilparikh2806/meta_hackathon_finals_hackwithpals/blob/main/fleet_train.ipynb)

---
*Made by Team HackWithPals | Meta Hackathon Finals 2026*
