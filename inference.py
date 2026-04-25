"""
Inference Script — AI Sprint Manager OpenEnv
============================================================
MANDATORY:
  API_BASE_URL  : LLM endpoint
  MODEL_NAME    : Model identifier
  HF_TOKEN      : Hugging Face / API key

Usage:
  python inference.py
"""
from __future__ import annotations
import os
import json
import time
import requests
from openai import OpenAI

# ── Config ────────────────────────────────────────────────────────────────────
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
API_KEY      = os.getenv("HF_TOKEN") or os.getenv("API_KEY", "dummy")
MODEL_NAME   = os.getenv("MODEL_NAME", "meta-llama/Llama-3.1-8B-Instruct")
ENV_BASE_URL = os.getenv("ENV_BASE_URL", "https://sejal-k-ai-sprint-manager.hf.space")

MAX_STEPS    = 12   # per episode
TEMPERATURE  = 0.2
MAX_TOKENS   = 300

TASKS = ["easy_sprint", "medium_sprint", "hard_sprint"]

client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

# ── Prompts ───────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are an expert Tech Lead managing an agile sprint.
Your goal: maximize task completion, balance developer workload, and meet deadlines.

Each step you must output a JSON action with this exact schema:
{
  "action_type": "<assign|reassign|reprioritize|unblock|skip>",
  "task_id": "<task id or null>",
  "dev_id": "<developer id or null>",
  "new_priority": <1-5 or null>
}

Rules:
- assign: put a backlog task onto an available developer
- reassign: move an in-progress task to a different developer  
- reprioritize: change a task's priority (1=highest, 5=lowest)
- unblock: unblock a blocked task
- skip: do nothing this step

Output ONLY the JSON object. No explanation, no markdown fences."""


def build_user_prompt(obs: dict) -> str:
    tasks_summary = []
    for t in obs["tasks"]:
        tasks_summary.append(
            f"  [{t['id']}] {t['name']} | type={t['task_type']} | "
            f"priority={t['priority']} | effort={t['effort']} | "
            f"deadline=day{t['deadline']} | status={t['status']} | "
            f"assigned={t['assigned_to']} | progress={t['progress']:.0%}"
        )

    devs_summary = []
    for d in obs["developers"]:
        devs_summary.append(
            f"  [{d['id']}] {d['name']} | skill={d['skill']} | "
            f"load={d['current_load']}/{d['capacity']} | "
            f"available={d['is_available']} | tasks={d['assigned_tasks']}"
        )

    events_str = "\n  ".join(obs.get("events", [])) or "None"

    return f"""=== SPRINT STATUS ===
Day: {obs['current_day']} / {obs['sprint_length']}
Completed: {obs['tasks_completed']} | Missed: {obs['tasks_missed']} | In Progress: {obs['tasks_in_progress']} | Backlog: {obs['tasks_backlog']}
Workload Balance: {obs['workload_balance_score']:.2f} (1=perfect)
Cumulative Reward: {obs['cumulative_reward']:.2f}

Recent Events:
  {events_str}

TASKS:
{chr(10).join(tasks_summary)}

DEVELOPERS:
{chr(10).join(devs_summary)}

What is your next action? Output only the JSON."""


def call_env(endpoint: str, payload: dict = None, method: str = "POST") -> dict:
    url = f"{ENV_BASE_URL}/{endpoint}"
    if method == "GET":
        resp = requests.get(url, timeout=30)
    else:
        resp = requests.post(url, json=payload or {}, timeout=30)
    resp.raise_for_status()
    return resp.json()


def parse_action(text: str) -> dict:
    """Extract JSON action from model response."""
    text = text.strip()
    # Strip markdown fences if present
    if "```" in text:
        lines = text.split("\n")
        lines = [l for l in lines if not l.strip().startswith("```")]
        text = "\n".join(lines)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Try to find first { ... } block
        start = text.find("{")
        end = text.rfind("}") + 1
        if start >= 0 and end > start:
            try:
                return json.loads(text[start:end])
            except json.JSONDecodeError:
                pass
    return {"action_type": "skip", "task_id": None, "dev_id": None, "new_priority": None}


def run_episode(task_name: str) -> float:
    """Run one complete episode and return final score."""
    print(f"\n{'='*60}")
    print(f"  Task: {task_name}")
    print(f"{'='*60}")

    # Reset
    obs = call_env("reset", {"task_name": task_name, "seed": 42})
    print(f"  Sprint started. Tasks: {obs['tasks_backlog']} | Devs: {len(obs['developers'])}")

    final_score = 0.0
    for step_num in range(1, MAX_STEPS + 1):
        if obs.get("done", False):
            break

        user_prompt = build_user_prompt(obs)

        try:
            completion = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": user_prompt},
                ],
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS,
            )
            response_text = completion.choices[0].message.content or ""
        except Exception as e:
            print(f"  [Step {step_num}] LLM error: {e}. Using skip.")
            response_text = '{"action_type": "skip"}'

        action = parse_action(response_text)
        print(f"  [Step {step_num}] Action: {action}")

        # Step environment
        result = call_env("step", {"action": action})
        obs = result["observation"]
        reward = result["reward"]
        done = result["done"]
        info = result.get("info", {})

        events = obs.get("events", [])
        if events:
            for ev in events:
                print(f"    → {ev}")
        print(f"    Reward: {reward:+.2f} | Cumulative: {obs['cumulative_reward']:.2f} | Done: {done}")

        if done:
            final_score = info.get("final_score", 0.0)
            break

    print(f"\n  ✅ Episode complete. Final score: {final_score:.4f}")
    return final_score


def main():
    print("\n🚀 AI Sprint Manager — Baseline Inference")
    print(f"   Model : {MODEL_NAME}")
    print(f"   Server: {ENV_BASE_URL}")

    # Health check
    try:
        health = call_env("health", method="GET")
        print(f"   Health: {health}")
    except Exception as e:
        print(f"   ⚠️  Could not reach env server at {ENV_BASE_URL}: {e}")
        print("   Make sure the server is running: uvicorn server:app --host 0.0.0.0 --port 8000")
        return

    scores = {}
    start_time = time.time()

    for task in TASKS:
        try:
            score = run_episode(task)
            scores[task] = score
        except Exception as e:
            print(f"  ❌ Error on {task}: {e}")
            scores[task] = 0.0

    elapsed = time.time() - start_time

    print(f"\n{'='*60}")
    print("  BASELINE SCORES")
    print(f"{'='*60}")
    for task, score in scores.items():
        bar = "█" * int(score * 20)
        print(f"  {task:<20} {score:.4f}  {bar}")
    avg = sum(scores.values()) / len(scores) if scores else 0.0
    print(f"  {'AVERAGE':<20} {avg:.4f}")
    print(f"\n  Runtime: {elapsed:.1f}s")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()