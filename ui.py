"""
Combined FastAPI + Gradio app for AI Sprint Manager
Runs everything on port 7860 for HF Spaces
"""
import gradio as gr
import requests
import json
from fastapi import FastAPI
from fastapi.responses import JSONResponse
import uvicorn

from sprint_env.environment import SprintManagerEnv
from sprint_env.models import SprintAction

# ── Single shared env instance ────────────────────────────────────────────────
env = SprintManagerEnv()

# ── FastAPI app ───────────────────────────────────────────────────────────────
api = FastAPI()

@api.post("/reset")
def reset(req: dict = {}):
    obs = env.reset(
        task_name=req.get("task_name", "easy_sprint"),
        seed=req.get("seed"),
        episode_id=req.get("episode_id"),
    )
    return obs.model_dump()

@api.post("/step")
def step(req: dict):
    action_data = req.get("action", {})
    action = SprintAction(**action_data)
    obs, reward, done, info = env.step(action)
    return {"observation": obs.model_dump(), "reward": reward, "done": done, "info": info}

@api.get("/state")
def state():
    return env.state.model_dump()

@api.get("/health")
def health():
    return {"status": "ok", "env": "ai-sprint-manager"}

@api.get("/tasks")
def tasks():
    return {"tasks": [
        {"id": "easy_sprint", "difficulty": "easy"},
        {"id": "medium_sprint", "difficulty": "medium"},
        {"id": "hard_sprint", "difficulty": "hard"},
    ]}

# ── Gradio UI functions (call env directly, not via HTTP) ─────────────────────

def reset_env(task_name):
    try:
        obs = env.reset(task_name=task_name, seed=42)
        obs_dict = obs.model_dump()
        return format_sprint_board(obs_dict), format_developers(obs_dict), format_events(obs_dict), obs_dict
    except Exception as e:
        return str(e), "", "", {}

def take_action(action_type, task_id, dev_id, new_priority, current_obs):
    try:
        action = SprintAction(
            action_type=action_type,
            task_id=task_id or None,
            dev_id=dev_id or None,
            new_priority=int(new_priority) if new_priority else None,
        )
        obs, reward, done, info = env.step(action)
        obs_dict = obs.model_dump()

        event_text = format_events(obs_dict)
        if reward != 0:
            event_text += f"\n💰 Step Reward: {reward:+.2f}"
        if done:
            event_text += f"\n\n🏁 SPRINT COMPLETE! Score: {info.get('final_score', 0):.2f}"

        return (
            format_sprint_board(obs_dict),
            format_developers(obs_dict),
            event_text,
            format_metrics(obs_dict),
            obs_dict,
        )
    except Exception as e:
        return str(e), "", "", "", current_obs

def format_sprint_board(obs):
    if not obs or "tasks" not in obs:
        return "No sprint active. Click Reset to start."

    sections = {"backlog": [], "in_progress": [], "done": [], "missed": [], "blocked": []}
    for t in obs["tasks"]:
        s = t["status"] if t["status"] in sections else "backlog"
        bar = "█" * int(t["progress"] * 10) + "░" * (10 - int(t["progress"] * 10))
        sections[s].append(
            f"  [{t['id']}] {t['name']}\n"
            f"       Type:{t['task_type']} | P{t['priority']} | "
            f"Effort:{t['effort']} | Due:Day{t['deadline']}\n"
            f"       Skill:{t['required_skill']} | Dev:{t['assigned_to'] or '-'}\n"
            f"       Progress: [{bar}] {t['progress']:.0%}"
        )

    day = obs.get("current_day", "?")
    sprint_len = obs.get("sprint_length", 10)
    day_bar = "▓" * int(day) + "░" * (int(sprint_len) - int(day))
    out = f"📅 Day {day}/{sprint_len}  [{day_bar}]\n"
    out += f"✅ Done:{obs['tasks_completed']}  ❌ Missed:{obs['tasks_missed']}  "
    out += f"🔄 In Progress:{obs['tasks_in_progress']}  📋 Backlog:{obs['tasks_backlog']}\n"
    out += "─" * 50 + "\n"

    for key, label in [("backlog","📋 BACKLOG"),("in_progress","🔄 IN PROGRESS"),
                        ("done","✅ DONE"),("missed","❌ MISSED"),("blocked","🚫 BLOCKED")]:
        if sections[key]:
            out += f"\n{label} ({len(sections[key])})\n"
            out += "\n".join(sections[key]) + "\n"
    return out

def format_developers(obs):
    if not obs or "developers" not in obs:
        return ""
    out = "👥 DEVELOPER WORKLOAD\n" + "─" * 40 + "\n"
    for d in obs["developers"]:
        load, cap = d["current_load"], d["capacity"]
        filled = min(int((load / cap) * 10) if cap > 0 else 0, 10)
        bar = "█" * filled + "░" * (10 - filled)
        status = "✅" if d["is_available"] else "🤒"
        out += f"{status} {d['name']} ({d['skill']})\n"
        out += f"   Load: [{bar}] {load}/{cap} pts\n"
        out += f"   Tasks: {d['assigned_tasks'] or 'none'}\n\n"
    return out

def format_events(obs):
    events = obs.get("events", [])
    return "\n".join(f"• {e}" for e in events) if events else "No events yet."

def format_metrics(obs):
    if not obs:
        return ""
    return (
        f"📊 Cumulative Reward: {obs.get('cumulative_reward', 0):.2f}\n"
        f"⚖️  Workload Balance: {obs.get('workload_balance_score', 0):.2f}/1.0\n"
        f"✅ Completed: {obs.get('tasks_completed', 0)}\n"
        f"❌ Missed: {obs.get('tasks_missed', 0)}"
    )

# ── Gradio UI ─────────────────────────────────────────────────────────────────
with gr.Blocks(title="🤖 AI Sprint Manager", theme=gr.themes.Soft()) as demo:
    current_obs = gr.State({})

    gr.Markdown("# 🤖 AI Sprint Manager — OpenEnv\n**RL environment for agile sprint management.**")

    with gr.Row():
        task_selector = gr.Dropdown(
            choices=["easy_sprint", "medium_sprint", "hard_sprint"],
            value="easy_sprint", label="Sprint Scenario"
        )
        reset_btn = gr.Button("🔄 Reset Sprint", variant="primary")

    with gr.Row():
        with gr.Column(scale=2):
            sprint_board = gr.Textbox(label="📋 Sprint Board", lines=25, interactive=False)
        with gr.Column(scale=1):
            dev_panel = gr.Textbox(label="👥 Developers", lines=12, interactive=False)
            metrics_panel = gr.Textbox(label="📊 Metrics", lines=6, interactive=False)

    gr.Markdown("### 🎮 Take Action")
    with gr.Row():
        action_type = gr.Dropdown(
            choices=["assign","reassign","reprioritize","unblock","skip"],
            value="assign", label="Action Type"
        )
        task_id_input = gr.Textbox(label="Task ID", placeholder="T1")
        dev_id_input = gr.Textbox(label="Developer ID", placeholder="dev1")
        priority_input = gr.Dropdown(choices=["","1","2","3","4","5"], value="", label="Priority")

    step_btn = gr.Button("▶️ Take Action", variant="primary")
    event_log = gr.Textbox(label="📜 Event Log", lines=5, interactive=False)

    gr.Markdown("""
    **Tips:** Backend tasks → Alice (dev1) | Frontend → Bob (dev2) | Any → Carol (dev3)
    """)

    reset_btn.click(reset_env, [task_selector], [sprint_board, dev_panel, event_log, current_obs])
    step_btn.click(take_action, [action_type, task_id_input, dev_id_input, priority_input, current_obs],
                   [sprint_board, dev_panel, event_log, metrics_panel, current_obs])

# ── Mount Gradio into FastAPI and serve everything on 7860 ────────────────────
app = gr.mount_gradio_app(api, demo, path="/")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7860)