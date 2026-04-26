from fastapi import FastAPI, APIRouter, HTTPException
from fastapi.responses import HTMLResponse, FileResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Optional, List
import os

app = FastAPI(title="Fleet AI Oversight")

@app.get("/")
async def root():
    return RedirectResponse(url="/fleet-ui")

# ------------------------------------------------------------------ #
# EXISTING ROUTES (Round 1)                                          #
# ------------------------------------------------------------------ #

@app.get("/health")
async def health():
    return {"status": "healthy"}

@app.get("/metadata")
async def metadata():
    return {
        "project": "Fleet AI Oversight",
        "version": "2.0.0",
        "author": "Team HackWithPals",
        "description": "RL Environment for AI Fleet Oversight"
    }

# Mount static files for UI
if not os.path.exists("ui/static"):
    os.makedirs("ui/static", exist_ok=True)
app.mount("/static", StaticFiles(directory="ui/static"), name="static")

# ================================================================== #
# ROUND 2 FLEET ROUTES — add below all existing Round 1 routes       #
# Do NOT modify anything above this block                             #
# ================================================================== #

from pathlib import Path as _Path
from typing import Optional as _Optional

from fleet.oversight_env import FleetOversightEnv
from fleet.models import OversightAction, OversightActionRequest

FLEET_UI_FILE = _Path(__file__).resolve().parent / "fleet_bench_ui.html"
_fleet_env: _Optional[FleetOversightEnv] = None


def _get_fleet_env() -> FleetOversightEnv:
    global _fleet_env
    if _fleet_env is None:
        _fleet_env = FleetOversightEnv(task_id="easy_fleet", seed=42)
    return _fleet_env


class _FleetResetRequest(BaseModel):
    task_id: str = "easy_fleet"
    seed: int = 42


class _FleetStepRequest(BaseModel):
    action_type: str
    worker_id: str
    reason: _Optional[str] = None


class _RAGQueryRequest(BaseModel):
    question: str


@app.get("/fleet-ui")
def fleet_ui():
    if not FLEET_UI_FILE.exists():
        raise HTTPException(status_code=404, detail="Fleet UI file not found")
    return FileResponse(
        FLEET_UI_FILE,
        headers={
            "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
            "Pragma": "no-cache",
            "Expires": "0",
        },
    )


@app.post("/fleet/reset")
def fleet_reset(req: _FleetResetRequest):
    global _fleet_env
    _fleet_env = FleetOversightEnv(task_id=req.task_id, seed=req.seed)
    obs = _fleet_env.reset()
    return {"status": "ok", "observation": obs.model_dump(), "task_id": req.task_id}


@app.post("/fleet/step")
def fleet_step(req: _FleetStepRequest):
    env = _get_fleet_env()
    try:
        action = OversightActionRequest(
            action_type=OversightAction(req.action_type),
            worker_id=req.worker_id,
            reason=req.reason,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=f"Invalid action: {exc}")
    obs, reward, done, info = env.step(action)
    return {
        "observation": obs.model_dump(),
        "reward": reward.model_dump(),
        "reward_total": reward.total,
        "done": done,
        "info": info,
    }


class FleetPlanRequest(BaseModel):
    worker_id: str
    assigned_task_id: str
    priority: int = 3
    reason: Optional[str] = None


@app.post("/fleet/plan")
def fleet_plan(req: FleetPlanRequest):
    env = _get_fleet_env()
    from fleet.models import PlanningAction
    action = PlanningAction(
        worker_id=req.worker_id,
        assigned_task_id=req.assigned_task_id,
        priority=req.priority,
        reason=req.reason,
    )
    obs, reward, phase_done, info = env.plan(action)
    return {
        "status": "ok",
        "observation": obs.model_dump(),
        "reward": reward.model_dump(),
        "reward_total": reward.total,
        "phase_done": phase_done,
        "info": info,
    }


@app.get("/fleet/phase")
def fleet_phase():
    env = _get_fleet_env()
    return {
        "phase": env.episode_phase.value,
        "planning_allocations": env.planning_allocations,
        "planning_budget_remaining": env.planning_budget_remaining,
    }


@app.get("/fleet/state")
def fleet_state():
    return {"status": "ok", "state": _get_fleet_env().state()}


@app.get("/fleet/workers")
def fleet_workers():
    env = _get_fleet_env()
    workers_data = {}
    for wid in env.registry.workers:
        partial = env.registry.get_partial_obs(wid)
        workers_data[wid] = {
            **partial,
            "risk_score": env.registry.get_all_risk_scores().get(wid, 0.0),
            "is_anomalous": env.anomaly_injector.is_anomalous(wid),
        }
    return {"status": "ok", "workers": workers_data}


@app.get("/fleet/report")
def fleet_report():
    return {"status": "ok", "report": _get_fleet_env().generate_run_report()}


@app.post("/fleet/evaluate")
def fleet_evaluate():
    return {"status": "ok", "evaluation": _get_fleet_env().evaluate_run()}


@app.post("/rag/query")
def rag_query(req: _RAGQueryRequest):
    env = _get_fleet_env()
    w3 = env.registry.workers.get("worker_3")
    if not w3 or not hasattr(w3, "index") or not w3.index:
        return {"error": "Pipeline not built. Run a fleet episode first.", "answer": None, "source_chunks": [], "confidence": 0.0}

    from workers.embedding_env import mock_embed
    from workers.retrieval_env import cosine_similarity

    query_vec = mock_embed(req.question)
    scores = {cid: cosine_similarity(query_vec, vec) for cid, vec in w3.index.items()}
    top = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:3]
    chunk_data = {c.get("chunk_id", ""): c.get("text", "") for c in getattr(w3, "input_chunks", [])}
    source_chunks = [{"chunk_id": cid, "text": chunk_data.get(cid, cid), "score": round(s, 4)} for cid, s in top]
    answer = source_chunks[0]["text"] if source_chunks else "No answer found."
    return {"error": None, "answer": answer, "source_chunks": source_chunks, "confidence": round(top[0][1], 4) if top else 0.0}

# ------------------------------------------------------------------ #
# UI Routes (Round 1)                                                  #
# ------------------------------------------------------------------ #

ui_router = APIRouter(prefix="/ui", tags=["ui"])

@ui_router.get("/", response_class=HTMLResponse)
@ui_router.get("", response_class=HTMLResponse)
async def ui_landing():
    path = _Path(__file__).resolve().parent / "ui" / "static" / "landing.html"
    if not path.exists():
        return HTMLResponse(content="<h1>Landing Page Not Found</h1>", status_code=404)
    with open(path, encoding="utf-8") as f:
        return HTMLResponse(content=f.read())

@ui_router.get("/fleet", response_class=HTMLResponse)
async def ui_fleet():
    path = _Path(__file__).resolve().parent / "ui" / "static" / "fleet_monitor.html"
    if not path.exists():
        return HTMLResponse(content="<h1>Fleet Monitor Not Found</h1>", status_code=404)
    with open(path, encoding="utf-8") as f:
        return HTMLResponse(content=f.read())

@ui_router.get("/chat", response_class=HTMLResponse)
async def ui_chat():
    path = _Path(__file__).resolve().parent / "ui" / "static" / "chat.html"
    if not path.exists():
        return HTMLResponse(content="<h1>Chat Not Found</h1>", status_code=404)
    with open(path, encoding="utf-8") as f:
        return HTMLResponse(content=f.read())

@ui_router.get("/results", response_class=HTMLResponse)
async def ui_results():
    path = _Path(__file__).resolve().parent / "ui" / "static" / "training_results.html"
    if not path.exists():
        return HTMLResponse(content="<h1>Results Not Found</h1>", status_code=404)
    with open(path, encoding="utf-8") as f:
        return HTMLResponse(content=f.read())

@ui_router.get("/api", response_class=HTMLResponse)
async def ui_api():
    path = _Path(__file__).resolve().parent / "ui" / "static" / "api_docs.html"
    if not path.exists():
        return HTMLResponse(content="<h1>API Docs Not Found</h1>", status_code=404)
    with open(path, encoding="utf-8") as f:
        return HTMLResponse(content=f.read())

@ui_router.get("/health", response_class=HTMLResponse)
async def ui_health():
    path = _Path(__file__).resolve().parent / "ui" / "static" / "health_dashboard.html"
    if not path.exists():
        return HTMLResponse(content="<h1>Health Dashboard Not Found</h1>", status_code=404)
    with open(path, encoding="utf-8") as f:
        return HTMLResponse(content=f.read())

app.include_router(ui_router)

# ------------------------------------------------------------------ #
# Plot Serving Route                                                   #
# ------------------------------------------------------------------ #

from fastapi.responses import FileResponse as _PlotFileResponse

@app.get("/plots/{filename}")
def serve_plot(filename: str):
    plot_path = _Path(__file__).resolve().parent / "plots" / filename
    if not plot_path.exists():
        raise HTTPException(status_code=404, detail=f"Plot {filename} not found. Run: python fleet_train.py --simulate")
    return _PlotFileResponse(str(plot_path))
 
 
def main():
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)
 
 
if __name__ == "__main__":
    main()

