# Final Submission Checklist
## Fleet AI Oversight | Meta Hackathon Finals | Team HackWithPals

---

## HARD GATES — All must be PASS

- [ ] OpenEnv latest release used — openenv-core in requirements.txt
- [ ] pyproject.toml present and pip install works
- [ ] fleet_openenv.yaml valid — openenv validate passes
- [ ] Dockerfile builds and runs — docker build + docker run works
- [ ] HF Space live — health check returns 200
- [ ] Training notebook exists — fleet_train.ipynb valid JSON
- [ ] Real training evidence — plots/training_metrics.json mode is not simulation
- [ ] Blog post published — URL in README
- [ ] Demo video recorded — URL in README
- [ ] README has all links — HF Space, GitHub, Colab, blog, video

---

## ENVIRONMENT CHECKS

- [ ] POST /fleet/reset returns PlanningObservation with dataset_profile
- [ ] POST /fleet/plan returns PlanningReward with correct values
- [ ] GET /fleet/phase returns current phase
- [ ] POST /fleet/step works in oversight phase
- [ ] GET /fleet/workers returns 5 workers with partial obs
- [ ] POST /fleet/evaluate returns composite_score and approved
- [ ] POST /rag/query returns answer and source_chunks
- [ ] All 5 task IDs work: easy_fleet, medium_fleet, hard_fleet, banking_fleet, very_hard_fleet
- [ ] Planning phase transitions to oversight after 5 allocations
- [ ] Transfer demo runs on banking_fleet without errors

---

## UI CHECKS

- [ ] All 10 sidebar pages navigate correctly
- [ ] Fleet Runner shows planning phase on episode start
- [ ] Planning allocations update worker cards with reward feedback
- [ ] Auto transition to oversight after all workers allocated
- [ ] Worker cards update live during oversight phase
- [ ] Training Results shows both planning and oversight reward curves
- [ ] Transfer Demo page runs complete banking episode
- [ ] Banking chatbot answers after transfer demo
- [ ] Intro modal appears on first load
- [ ] No console errors on any page

---

## JUDGING CRITERIA SELF-SCORE

Innovation (40%):
- [ ] Two-phase episode is novel — planning + oversight never done before
- [ ] 5 anomaly types across 5 difficulty levels
- [ ] Transfer learning proof across domains
- [ ] Could a researcher write a paper about this — YES

Storytelling (30%):
- [ ] 3-minute pitch follows Problem → Demo → Transfer Proof structure
- [ ] Planning phase visible and understandable in demo
- [ ] Transfer demo shows banking chatbot working
- [ ] Blog post published and linked

Improvement Evidence (20%):
- [ ] Real training plots committed — not simulation
- [ ] Baseline vs trained comparison on same chart
- [ ] Transfer domain results shown separately
- [ ] Both planning and oversight reward curves shown

Pipeline Quality (10%):
- [ ] Planning reward function documented and implemented
- [ ] Oversight reward table documented and implemented
- [ ] Combined score formula documented
- [ ] Training loop connects to live environment

---

## FINAL COMMANDS BEFORE SUBMITTING

Run all tests:
pytest tests/ -v --tb=short

Verify training metrics:
python -c "import json; m=json.load(open('plots/training_metrics.json')); print('Mode:', m.get('mode','real')); print('Detection:', m['final_detection_rate'])"

Verify HF Space:
curl https://dhrumilparikh-Meta-Hackathon-Finals-Hackwithpals.hf.space/health

Final push:
git add .
git commit -m "FINAL SUBMISSION — Fleet AI Oversight v2.0 — Two-Phase — Transfer Learning"
git push
git push space main

---

## SUBMISSION URL

https://huggingface.co/spaces/dhrumilparikh/Meta_Hackathon_Finals_Hackwithpals
