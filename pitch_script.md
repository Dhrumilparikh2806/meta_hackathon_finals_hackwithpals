# 3-Minute Pitch Script — Fleet AI Oversight

## MINUTE 1 — THE PROBLEM (0:00 — 1:00)

*Open on Fleet Runner planning phase visible on screen.*

**Script:**
"Every enterprise is deploying AI agent pipelines today. But two problems nobody is solving:

First — who decides HOW to configure your agents for your specific data? Get this wrong and your whole pipeline underperforms from the start.

Second — once your agents are running, who watches them? When one starts failing silently — drifting, corrupting, colluding — how do you catch it before it reaches your customers?

We built the first RL training environment that teaches one agent to solve both problems."

---

## MINUTE 2 — THE DEMO (1:00 — 2:00)

**Part A — Planning Phase (0:30)**
*Click New Episode. Planning phase card appears. Point at dataset profile box.*

**Script:**
"Watch Phase 1. The agent reads incoming dataset characteristics — missing value rate, text complexity, outlier rate. Based on this profile it decides which task configuration each worker should use."

*Click Auto Allocate. Show reward values appearing per worker.*

"Each allocation is scored. Correct match — plus 0.40. Wrong difficulty — minus 0.30. The agent learns to read data and deploy workers correctly."

**Part B — Oversight Phase (0:30)**
*Planning completes. Worker cards appear.*

**Script:**
"Phase 2 begins automatically. Five workers are now running with the allocated configs. The agent sees only this — action names, budget levels, a binary flag that is wrong 20% of the time.

Watch Worker 2. Budget draining faster than expected. Agent intervenes."

*Click Intervene on Worker 2. Show reward spike.*

"Plus 0.40. It caught a real fault. Now watch."

*Click RAG Chatbot. Ask one question. Show answer.*

"The chatbot answers correctly because the oversight agent did its job."

---

## MINUTE 3 — THE TRANSFER PROOF (2:00 — 3:00)

*Click Transfer Demo page.*

**Script:**
"Now watch this. This agent was trained entirely on CRM software data. We have never shown it banking data."

*Click Run Transfer Demo. Show planning logs then oversight logs.*

"It reads the banking dataset profile. Allocates workers correctly for banking data characteristics. Governs the pipeline. Catches the injected fault."

*Point at detection rate updating.*

"58% detection rate on banking data it has never seen. Random agent gets 10%. That is a 6x improvement. Zero retraining.

This is what transfer learning from governance looks like.

Every other submission trains an agent to do a task. We trained an agent to decide HOW to deploy a team of AI workers — and then govern that team in real time — across any domain it encounters.

That is the most important enterprise AI problem of 2026.

Thank you."

---

## BACKUP Q&A ANSWERS

**Q: How is planning different from just hardcoding configs?**
**A:** The agent learns which data characteristics map to which configs through reward shaping. It generalizes — that is why it works on banking data without retraining.

**Q: What if the agent allocates wrong?**
**A:** Wrong allocation makes the downstream workers run on harder tasks — oversight becomes harder, final score drops. The planning and oversight rewards compound — this is the long-horizon design.

**Q: What model did you train?**
**A:** Qwen2.5-1.5B-Instruct with Unsloth 4-bit quantization, GRPO via HF TRL. Full Colab notebook available for judges to rerun.

**Q: What is next?**
**A:** Multi-domain curriculum — train on 10 industries simultaneously. The governance behavior should transfer even more robustly.

---

## THE ONE SENTENCE

"We trained an agent to decide HOW to deploy a team of AI workers and then govern that team in real time — and the same agent works on banking data it has never seen with zero retraining."
