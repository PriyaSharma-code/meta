"""
OpenEnv server entry point for ai-sprint-manager.
"""
import uvicorn
from fastapi import FastAPI
from sprint_env.environment import SprintManagerEnv
from sprint_env.models import SprintAction

env = SprintManagerEnv()

app = FastAPI(
    title="AI Sprint Manager — OpenEnv",
    version="1.0.0",
)

@app.post("/reset")
def reset(req: dict = {}):
    obs = env.reset(
        task_name=req.get("task_name", "easy_sprint"),
        seed=req.get("seed"),
        episode_id=req.get("episode_id"),
    )
    return obs.model_dump()

@app.post("/step")
def step(req: dict):
    action = SprintAction(**req.get("action", {}))
    obs, reward, done, info = env.step(action)
    return {"observation": obs.model_dump(), "reward": reward, "done": done, "info": info}

@app.get("/state")
def state():
    return env.state.model_dump()

@app.get("/health")
def health():
    return {"status": "ok", "env": "ai-sprint-manager"}

@app.get("/tasks")
def tasks():
    return {"tasks": [
        {"id": "easy_sprint", "difficulty": "easy"},
        {"id": "medium_sprint", "difficulty": "medium"},
        {"id": "hard_sprint", "difficulty": "hard"},
    ]}

def main():
    uvicorn.run(app, host="0.0.0.0", port=8000)

if __name__ == "__main__":
    main()