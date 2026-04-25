"""
train_llm.py — AI Sprint Manager R1+R2 Training
================================================
TRAINING APPROACH: SFT warm-up → GRPO fine-tuning (curriculum)

═══════════════════════════════════════════════════════════════
WHY THIS COMBINATION?
═══════════════════════════════════════════════════════════════

1. SFT WARM-UP (Phase 0, optional but recommended)
   ─────────────────────────────────────────────────
   Why: Cold-start GRPO on Llama-3.1-8B with a random policy can collapse
   early because all 4 generations get similar rewards → GRPO gradient is
   zero → no learning signal. A brief SFT warm-up (1–2 epochs on rule-based
   trajectories) teaches the model the output FORMAT (JSON schema) before
   reward-driven exploration begins.

   What we SFT on: (observation, rule-based_action) pairs. Rule-based is
   NOT the optimal policy — it's just good enough to seed valid JSON output.

   This is identical to how InstructGPT (and DeepSeek-R1) bootstraps: SFT
   first for format, then RLHF/GRPO for quality.

2. GRPO FINE-TUNING (Main training)
   ──────────────────────────────────
   Why GRPO over PPO: No value network required → 40% less GPU memory.
   On a T4 (16GB) with 4-bit QLoRA this is the difference between fitting
   and OOMing.

   Why GRPO over RLVR/RLVE: RLVR (RL with Verifiable Rewards) applies when
   the reward is binary (correct/incorrect), e.g. maths problems. Our reward
   is continuous and multi-component → GRPO with group-relative normalisation
   is a better fit.

   GRPO mechanism:
     - Sample num_generations=4 completions per prompt
     - Each completion is an action JSON
     - Call /step on the env → get step_reward
     - Compute group baseline = mean(rewards over 4 generations)
     - Policy gradient = encourage completions above baseline, penalise below
     - KL divergence penalty (beta) prevents policy from drifting too far
       from the reference model (anti-reward-hacking measure #1)

3. CURRICULUM LEARNING (Phase both)
   ──────────────────────────────────
   Ratio 2:2 (R1 then R2) per group of 4. R1 tasks (10 steps, simple) give
   a denser reward signal early in training; R2 tasks (60 steps, complex)
   build long-horizon planning. Without curriculum, GRPO on R2 cold-start
   is extremely sample-inefficient.

═══════════════════════════════════════════════════════════════
REWARD DESIGN & ANTI-HACKING MEASURES
═══════════════════════════════════════════════════════════════

R1 REWARD (single sprint, 10 days):
  step_reward from /step:
    +1.5 to +2.0  : assign task → task completes by deadline
    +0.5 to +1.0  : correct skill match
    -0.05         : skip (opportunity cost, not catastrophic)
    -0.1          : assign already-in_progress task (invalid)
    -2.0 to -2.5  : sprint ends with incomplete high-priority task
    -0.2          : unknown action type

  Normalised for GRPO:
    r_norm = clip((step_reward + 3.0) / 5.0, 0, 1)
    Shift +3 centres the no-op case at 0.60 so the model sees a gradient
    even for neutral actions. Without this shift, most rewards cluster near
    0.0 and GRPO training collapses.

  ANTI-HACKING: R1 is graded by the graders.py functions which are
  SERVER-SIDE and stateful. The agent cannot fake a task completion —
  the server checks effort remaining, developer availability, deadlines.

R2 REWARD (multi-sprint, 60 days):
  Three-level reward structure:
    a) Step reward (dense, every action):
         Same structure as R1 step reward.
    b) Sprint-boundary bonus (every 10 days):
         +0.5 per sprint completed above threshold delivery rate.
         -0.3 per developer with burnout (productivity < 0.5).
         -2.0 per task missed at sprint boundary.
    c) Final project score (day 60 only, sparse):
         delivery_rate = tasks_completed / tasks_total
         team_health   = max(0.01, 1.0 - tech_debt_items * 0.02)
         final_score   = delivery_rate * 0.55
                       + instruction_following_score * 0.30
                       + team_health * 0.15

  Combined training reward:
    step_norm  = clip((step_reward + 3.0) / 5.0, 0, 1)
    combined   = step_norm * 0.6 + inst_score * 0.4

  Why instruction_following_score as auxiliary reward?
    - Without it, GRPO learns to ignore instructions (they're sparse signals).
    - inst_score is a running average: 1.0 if agent always acts on active
      instructions, 0.0 if it always ignores them.
    - Adding it as 0.4 weight makes every step instruction-aware.

  ANTI-HACKING MEASURES:
    1. KL penalty (beta=0.04): penalises policy that diverges too far from
       base model — prevents the model from finding degenerate strategies
       like outputting skip forever.
    2. Reward normalisation per GRPO group: absolute magnitudes don't matter,
       only relative ordering within each batch → can't inflate reward by
       gaming normalisation scale.
    3. Server-side state: the environment server is authoritative. The reward
       function can't be gamed client-side because:
         - task completion requires server-tracked effort countdown
         - instruction_following_score computed by server from actual actions
         - tech_debt is server state, can't be cleared by client action
    4. Clamping: all scores clamped to [0.01, 0.99] — the model can't earn
       a 1.0 reward by any single action, so it can't learn a trivial hack.
    5. Episode-reset: each reward_fn call resets the environment to a fixed
       seed. The model cannot carry state between reward evaluations.
    6. Skip penalty (-0.05): prevents "always skip" degenerate policy since
       skipping looks cheaper than risking a wrong assignment. The penalty
       makes skipping costlier than a bad assignment in the medium term.
    7. tech_debt permanent drag: each missed sprint task permanently reduces
       a developer's productivity by 2%. This makes short-horizon hacking
       (rush tasks to get early reward) self-defeating over 60 days.

FIXES IN THIS VERSION vs. original train_llm.py:
  [FIX-T1] reward_fn now resets env BEFORE calling /step (was calling /step
            on stale state from a previous episode — reward was meaningless).
  [FIX-T2] Observation extracted from reset response, not from step (reset
            returns obs directly, not wrapped in observation key).
  [FIX-T3] SFT warmup phase added (--phase sft or --sft-epochs > 0).
  [FIX-T4] Dataset now samples from MIDDLE of episodes (steps 3-8) not just
            from step 0 — gives the model harder, more diverse states to learn
            from. Step-0 prompts are trivially easy and over-represented.
  [FIX-T5] Tokenizer pad_token fix: Llama has no pad_token by default →
            setting to eos_token (standard practice).
  [FIX-T6] GRPOConfig: removed unsupported fields for older trl versions;
            added graceful version detection.
  [FIX-T7] Push uses model.merge_and_unload() before push if Unsloth so
            the pushed model is a full weight checkpoint, not just LoRA diff.
  [FIX-T8] Neutral fallback reward changed from 0.3 to 0.5 (true neutral)
            so env failures don't bias toward low-reward actions.
  [FIX-T9] build_grpo_dataset wraps each episode in try/except so a single
            HF Space timeout doesn't kill the whole dataset collection.

═══════════════════════════════════════════════════════════════
RECOMMENDED GPU USAGE
═══════════════════════════════════════════════════════════════
Model: Qwen2.5-1.5B for TRAINING (loaded locally — no HF router needed).
       Llama-3.1-8B for INFERENCE via HF router.

| GPU  | VRAM | Batch | Generations | Approx time (300ep) |
|------|------|-------|-------------|---------------------|
| T4   | 16GB | 1     | 2           | ~4-5 hours          |
| A10G | 24GB | 2     | 4           | ~2-3 hours          |
| A100 | 40GB | 4     | 4           | ~60-90 min          |

Colab setup (9 cells) — see PROJECT_HANDOFF.md for details.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Optional

# ── Env config ────────────────────────────────────────────────────────────────

ENV_BASE_URL = os.getenv("ENV_BASE_URL", "https://sejal-k-ai-sprint-manager.hf.space")
HF_TOKEN     = os.getenv("HF_TOKEN", "")
MODEL_NAME   = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-1.5B-Instruct")  # for LOCAL training
HF_REPO_ID   = os.getenv("HF_REPO_ID", "")

# ── GRPO hyperparameters ───────────────────────────────────────────────────────
# Defaults for T4 (16GB). Adjust per GPU via CLI or env override.
GRPO_CONFIG = {
    "learning_rate":               5e-6,
    "num_train_epochs":            1,
    "per_device_train_batch_size": 1,   # T4-safe default; double for A10G+
    "gradient_accumulation_steps": 8,   # effective batch=8 even with bs=1
    "max_prompt_length":           1024,
    "max_completion_length":       96,  # action JSON < 80 tokens; 96 gives margin
    "num_generations":             2,   # T4-safe; set 4 for A10G+
    "temperature":                 0.8, # high: encourage diverse candidates
    "beta":                        0.04,  # KL penalty — anti-reward-hacking
    "logging_steps":               5,
    "save_steps":                  50,
    "warmup_ratio":                0.05,
    "seed":                        42,
}

# SFT warm-up config
SFT_CONFIG = {
    "num_train_epochs":            2,
    "per_device_train_batch_size": 2,
    "gradient_accumulation_steps": 4,
    "learning_rate":               2e-5,  # higher than GRPO — SFT is supervised
    "warmup_ratio":                0.05,
    "logging_steps":               5,
    "save_steps":                  100,
}

R1_TASKS = ["easy_sprint", "medium_sprint", "hard_sprint"]
R2_TASKS = ["project_easy", "project_medium", "project_hard"]

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)


# ── System prompts ────────────────────────────────────────────────────────────
# Must match inference_r2.py exactly so training and inference are aligned.

R1_SYSTEM_PROMPT = """You are an expert Tech Lead managing an agile sprint.
Each step output a JSON action with this exact schema:
{"action_type":"<assign|reassign|reprioritize|unblock|skip>","task_id":"<id or null>","dev_id":"<id or null>","new_priority":<1-5 or null>}
Rules:
- assign: backlog task onto available developer (prefer skill match)
- reassign: move in-progress task to different developer
- reprioritize: change priority (1=highest)
- unblock: only for BLOCKED tasks (not backlog)
- skip: do nothing
Output ONLY the JSON. No explanation."""

R2_SYSTEM_PROMPT = """You are an Engineering Manager running a 60-day software project.
Each step you MUST output exactly ONE JSON object and nothing else.

Schema (use null for unused fields):
{"action_type":"<assign|reassign|reprioritize|unblock|skip>","task_id":"<id or null>","dev_id":"<id or null>","new_priority":<1-5 or null>}

Rules (follow in order):
1. If ACTIVE INSTRUCTIONS exist, assign THEIR tasks first.
2. Only assign tasks with status=backlog (never in_progress or done).
3. Only assign if all dependency markers show ✓.
4. Only assign to an AVAILABLE developer with matching or fullstack skill.
5. Use unblock ONLY for explicitly blocked tasks whose deps are ✓.
6. skip is last resort.

Output ONLY the JSON. No explanation."""


# ── Rule-based fallback policies ─────────────────────────────────────────────

def smart_fallback_r1(obs: dict) -> dict:
    """R1: assign highest-priority backlog task with skill match."""
    tasks  = obs.get("tasks", [])
    devs   = obs.get("developers", [])
    avail  = [d for d in devs
              if d.get("is_available", False) and d.get("current_load", 0) < d.get("capacity", 5)]
    backlog = sorted(
        [t for t in tasks if t.get("status") == "backlog"],
        key=lambda t: (t.get("priority", 9), t.get("deadline", 99))
    )
    for task in backlog:
        skill = task.get("required_skill", "")
        match = [d for d in avail if d.get("skill") == skill or d.get("skill") == "fullstack"]
        dev   = match[0] if match else (avail[0] if avail else None)
        if dev:
            return {"action_type": "assign", "task_id": task["id"],
                    "dev_id": dev["id"], "new_priority": None}
    return {"action_type": "skip", "task_id": None, "dev_id": None, "new_priority": None}


def smart_fallback_r2(obs: dict, assigned_set: Optional[set] = None) -> dict:
    """
    R2: instruction-first, dep-aware, skill-matching fallback.
    assigned_set prevents re-assigning tasks already assigned this episode.
    """
    if assigned_set is None:
        assigned_set = set()

    tasks     = obs.get("tasks", [])
    devs      = obs.get("developers", [])
    done_ids  = {t["id"] for t in tasks if t.get("status") == "done"}
    available = [d for d in devs
                 if d.get("is_available", False)
                 and d.get("current_load", 0) < d.get("capacity", 5) * 2]

    def best_dev(task: dict) -> Optional[dict]:
        skill = task.get("required_skill", "")
        match = [d for d in available if d.get("skill") == skill or d.get("skill") == "fullstack"]
        return match[0] if match else (available[0] if available else None)

    def can_assign(task: dict) -> bool:
        if task.get("status") != "backlog":
            return False
        if task["id"] in assigned_set:
            return False
        deps = task.get("metadata", {}).get("depends_on", [])
        return all(d in done_ids for d in deps)

    skip = {"action_type": "skip", "task_id": None, "dev_id": None, "new_priority": None}

    # 1. Instructions first
    active = sorted(
        [i for i in obs.get("instruction_queue", []) if not i.get("followed", False)],
        key=lambda i: i.get("target_sprint", 99)
    )
    for inst in active:
        for tid in inst.get("affects_tasks", []):
            task = next((t for t in tasks if t["id"] == tid), None)
            if task and can_assign(task):
                dev = best_dev(task)
                if dev:
                    return {"action_type": "assign", "task_id": task["id"],
                            "dev_id": dev["id"], "new_priority": None}

    # 2. Highest-priority backlog with deps met
    backlog = sorted(
        [t for t in tasks if t.get("status") == "backlog"],
        key=lambda t: (t.get("priority", 9), t.get("deadline", 99))
    )
    for task in backlog:
        if can_assign(task):
            dev = best_dev(task)
            if dev:
                return {"action_type": "assign", "task_id": task["id"],
                        "dev_id": dev["id"], "new_priority": None}

    # 3. Unblock
    for task in tasks:
        if task.get("status") == "blocked":
            deps = task.get("metadata", {}).get("depends_on", [])
            if all(d in done_ids for d in deps):
                return {"action_type": "unblock", "task_id": task["id"],
                        "dev_id": None, "new_priority": None}

    return skip


# ── Action parser ─────────────────────────────────────────────────────────────

_VALID_ACTIONS = {"assign", "reassign", "reprioritize", "skip", "unblock"}
_NULL_STRINGS  = {"null", "none", "None", "Null", "", "undefined", "N/A", "nil"}


def _parse_action(text: str) -> dict:
    """
    Parse LLM completion → action dict.
    Takes the LAST JSON object in the text (handles chain-of-thought prefix).
    """
    text = text.strip()
    if "```" in text:
        text = "\n".join(l for l in text.split("\n") if not l.strip().startswith("```"))

    # Find the LAST balanced {...} block
    d          = None
    depth      = 0
    obj_start  = -1
    last_start = -1
    last_end   = -1
    for i, ch in enumerate(text):
        if ch == "{":
            if depth == 0:
                obj_start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and obj_start >= 0:
                last_start = obj_start
                last_end   = i + 1

    if last_start >= 0:
        try:
            d = json.loads(text[last_start:last_end])
        except json.JSONDecodeError:
            pass
    if d is None:
        try:
            d = json.loads(text)
        except Exception:
            pass
    if d is None:
        return {"action_type": "skip", "task_id": None, "dev_id": None, "new_priority": None}

    raw = str(d.get("action_type", "skip")).lower().strip()
    d["action_type"] = raw if raw in _VALID_ACTIONS else "skip"

    for key in ("task_id", "dev_id", "new_priority"):
        val = d.get(key)
        if val is not None and str(val).strip() in _NULL_STRINGS:
            d[key] = None

    if d.get("new_priority") is not None:
        try:
            p = int(d["new_priority"])
            d["new_priority"] = p if 1 <= p <= 5 else None
        except (ValueError, TypeError):
            d["new_priority"] = None

    atype = d["action_type"]
    if atype in ("assign", "reassign") and (not d.get("task_id") or not d.get("dev_id")):
        d["action_type"] = "skip"
    if atype == "reprioritize" and (not d.get("task_id") or d.get("new_priority") is None):
        d["action_type"] = "skip"
    if atype == "unblock" and not d.get("task_id"):
        d["action_type"] = "skip"

    return {"action_type": d["action_type"], "task_id": d.get("task_id"),
            "dev_id": d.get("dev_id"), "new_priority": d.get("new_priority")}


# ── Prompt builders ───────────────────────────────────────────────────────────

def _build_r1_prompt(obs: dict) -> str:
    tasks_summary = "\n".join(
        f"  [{t['id']}] {t.get('name','?')} | P{t.get('priority','?')} | effort={t.get('effort','?')} "
        f"| due=Day{t.get('deadline','?')} | status={t.get('status','?')} | dev={t.get('assigned_to','none')}"
        for t in obs.get("tasks", [])
    )
    devs_summary = "\n".join(
        f"  [{d['id']}] {d.get('name','?')} | skill={d.get('skill','?')} "
        f"| load={d.get('current_load',0)}/{d.get('capacity',5)} | avail={d.get('is_available',False)}"
        for d in obs.get("developers", [])
    )
    return (
        f"Day: {obs.get('current_day',1)}/{obs.get('sprint_length',10)}\n"
        f"Done:{obs.get('tasks_completed',0)} Missed:{obs.get('tasks_missed',0)} "
        f"InProgress:{obs.get('tasks_in_progress',0)} Backlog:{obs.get('tasks_backlog',0)}\n"
        f"Cumulative Reward: {obs.get('cumulative_reward',0):.2f}\n\n"
        f"TASKS:\n{tasks_summary}\n\nDEVELOPERS:\n{devs_summary}\n\n"
        f"Output your JSON action:"
    )


def _build_r2_prompt(obs: dict) -> str:
    """Compact R2 prompt matching inference_r2.py format."""
    current_sprint = obs.get("current_sprint", 1)
    current_day    = obs.get("current_day", 1)
    days_left      = max(0, current_sprint * 10 - current_day + 1)
    tasks     = obs.get("tasks", [])
    done_ids  = {t["id"] for t in tasks if t.get("status") == "done"}

    active_insts = [i for i in obs.get("instruction_queue", []) if not i.get("followed", False)]
    inst_section = (
        "⚡FOLLOW: " + " | ".join(f"[{i['id']}] {i['text'][:50]}" for i in active_insts[:2])
    ) if active_insts else "No instructions."

    debt_count = len(obs.get("tech_debt", []))
    backlog = sorted([t for t in tasks if t.get("status") == "backlog"],
                     key=lambda t: (t.get("priority", 9), t.get("deadline", 99)))
    in_prog = [t for t in tasks if t.get("status") == "in_progress"]

    def fmt(t: dict) -> str:
        deps = t.get("metadata", {}).get("depends_on", [])
        dep_ok = "✓" if all(d in done_ids for d in deps) else "✗"
        return f"[{t['id']}]P{t.get('priority','?')} {str(t.get('required_skill','?'))[:4]} {dep_ok} D{t.get('deadline','?')}"

    backlog_str = " ".join(fmt(t) for t in backlog[:6])
    if len(backlog) > 6:
        backlog_str += f" +{len(backlog)-6}"
    inprog_str = " ".join(f"[{t['id']}]→{t.get('assigned_to','?')}" for t in in_prog) or "none"
    avail_devs = [d for d in obs.get("developers", []) if d.get("is_available", False)]
    devs_str   = " ".join(
        f"[{d['id']}]{str(d.get('name','?'))[:4]}({str(d.get('skill','?'))[:3]}) "
        f"{d.get('current_load',0)}/{d.get('capacity',5)}"
        for d in avail_devs
    )

    return (
        f"D{current_day}/60 S{current_sprint}/6 {days_left}d "
        f"done={obs.get('tasks_completed',0)} miss={obs.get('tasks_missed',0)} "
        f"inst={obs.get('instruction_following_score',0):.2f} debt={debt_count}\n"
        f"{inst_section}\n"
        f"BACKLOG(✓=deps_ok): {backlog_str}\n"
        f"IN_PROG: {inprog_str}\n"
        f"DEVS(avail): {devs_str}\n"
        f"JSON:"
    )


# ── GRPO reward functions ─────────────────────────────────────────────────────
#
# ANTI-HACKING design:
#   - Each reward_fn call resets the environment to a FIXED seed before stepping.
#     The model cannot exploit carry-over state between generations.
#   - Reward is normalised within the GRPO group (not absolute), so inflating
#     one generation's reward doesn't help unless it's relatively better.
#   - KL penalty (beta=0.04) prevents the policy from drifting to degenerate
#     strategies (e.g. "always assign T01 to dev1" for a cheap reward spike).
#   - Neutral fallback = 0.5 (true neutral), not 0.3, so env failures don't
#     bias learning toward low-reward actions.

def make_reward_fn(env_base_url: str, phase: str):
    """
    Returns a GRPO reward function evaluating completions against the live env.

    [FIX-T1] Resets env immediately before stepping — the env MUST be in the
    correct initial state when we evaluate each action. Previously, the reset
    was called but the obs was ignored and /step ran on whatever the env's
    current state was (potentially mid-episode from a previous call).
    """
    import requests

    episode_counter = [0]

    def _post(url: str, payload: dict) -> dict:
        resp = requests.post(url, json=payload, timeout=60)
        resp.raise_for_status()
        return resp.json()

    def reward_fn(prompts, completions, **kwargs) -> list[float]:
        rewards = []
        for prompt, completion in zip(prompts, completions):
            episode_counter[0] += 1
            n = episode_counter[0]

            # Curriculum: alternating R1/R2 (2:2 ratio for "both" phase)
            if phase == "r1":
                use_r2 = False
            elif phase == "r2":
                use_r2 = True
            else:
                use_r2 = (n % 4 >= 2)

            action = _parse_action(completion)
            r = 0.5   # true-neutral fallback [FIX-T8]

            try:
                if not use_r2:
                    # [FIX-T1] Reset, then step
                    task = R1_TASKS[n % len(R1_TASKS)]
                    _post(f"{env_base_url}/reset",
                          {"task_name": task, "seed": n % 100})
                    result = _post(f"{env_base_url}/step", {"action": action})

                    step_r = float(result.get("reward", 0.0))
                    # Normalise R1 reward: [-3,+2] → [0,1]
                    r = max(0.0, min(1.0, (step_r + 3.0) / 5.0))

                else:
                    # [FIX-T1] Reset project, then step
                    task = R2_TASKS[n % len(R2_TASKS)]
                    _post(f"{env_base_url}/project/reset",
                          {"task_name": task, "seed": n % 100})
                    result = _post(f"{env_base_url}/project/step",
                                   {"action": action})

                    step_r     = float(result.get("reward", 0.0))
                    obs2       = result.get("observation", {})
                    inst_score = float(obs2.get("instruction_following_score", 0.5))
                    step_norm  = max(0.0, min(1.0, (step_r + 3.0) / 5.0))

                    # Combined: step quality (60%) + instruction compliance (40%)
                    # Anti-hacking: inst_score is server-side computed running avg;
                    # can't be faked by the model outputting anything in particular.
                    r = step_norm * 0.6 + inst_score * 0.4

            except Exception as e:
                print(f"[WARN] reward_fn env call failed: {e}", flush=True)
                r = 0.5  # [FIX-T8] true neutral

            rewards.append(float(r))
        return rewards

    return reward_fn


# ── Dataset builder ───────────────────────────────────────────────────────────
#
# [FIX-T4] Samples from MIDDLE of episodes (skip first N steps).
# The first 1-2 steps are trivially easy (full backlog, no instructions yet).
# Sampling from steps 3+ gives the model harder, more diverse training states.
# [FIX-T9] Each episode wrapped in try/except — HF Space timeouts don't kill collection.

def build_grpo_dataset(n_examples: int = 200, phase: str = "both"):
    """
    Build a HuggingFace Dataset of (prompt) examples.
    Each prompt is a serialised chat message list.
    The LLM is NOT called — only rule-based policy advances the game.
    """
    try:
        from datasets import Dataset
    except ImportError:
        print("[ERROR] datasets not installed. Run: pip install datasets", flush=True)
        sys.exit(1)

    import requests

    def post(url: str, payload: dict) -> dict:
        r = requests.post(url, json=payload, timeout=60)
        r.raise_for_status()
        return r.json()

    examples = []
    tasks_r1 = R1_TASKS if phase in ("r1", "both") else []
    tasks_r2 = R2_TASKS if phase in ("r2", "both") else []
    per_task = max(1, n_examples // max(1, len(tasks_r1) + len(tasks_r2)))

    SKIP_STEPS_R1 = 1   # skip first step (trivial full-backlog state)
    SKIP_STEPS_R2 = 2   # skip first 2 steps (instructions not yet released)
    SAMPLE_PER_EP = 6   # states to sample per episode

    # ── Collect R1 snapshots ──────────────────────────────────────────────────
    for task_name in tasks_r1:
        print(f"  [DATASET] R1 {task_name} × {per_task} episodes...", flush=True)
        for ep in range(per_task):
            try:
                obs = post(f"{ENV_BASE_URL}/reset", {"task_name": task_name, "seed": ep})
                # [FIX-T4] Advance past trivial early steps
                for _ in range(SKIP_STEPS_R1):
                    if obs.get("done", False):
                        break
                    action = smart_fallback_r1(obs)
                    result = post(f"{ENV_BASE_URL}/step", {"action": action})
                    obs    = result.get("observation", obs)
                    if result.get("done", False):
                        break

                for step in range(SAMPLE_PER_EP):
                    if obs.get("done", False):
                        break
                    prompt = _build_r1_prompt(obs)
                    examples.append({
                        "prompt": [
                            {"role": "system", "content": R1_SYSTEM_PROMPT},
                            {"role": "user",   "content": prompt},
                        ],
                    })
                    action = smart_fallback_r1(obs)
                    result = post(f"{ENV_BASE_URL}/step", {"action": action})
                    obs    = result.get("observation", obs)
                    if result.get("done", False):
                        break
            except Exception as e:
                print(f"  [WARN] R1 ep{ep} failed: {e}", flush=True)  # [FIX-T9]

    # ── Collect R2 snapshots ──────────────────────────────────────────────────
    for task_name in tasks_r2:
        print(f"  [DATASET] R2 {task_name} × {per_task} episodes...", flush=True)
        for ep in range(per_task):
            try:
                obs = post(f"{ENV_BASE_URL}/project/reset",
                           {"task_name": task_name, "seed": ep})
                assigned_set: set[str] = set()

                # [FIX-T4] Advance past trivial initial state
                for _ in range(SKIP_STEPS_R2):
                    if obs.get("done", False):
                        break
                    action = smart_fallback_r2(obs, assigned_set)
                    if action["action_type"] == "assign" and action.get("task_id"):
                        assigned_set.add(action["task_id"])
                    result = post(f"{ENV_BASE_URL}/project/step", {"action": action})
                    obs    = result.get("observation", obs)
                    if result.get("done", False):
                        break

                for step in range(SAMPLE_PER_EP):
                    if obs.get("done", False):
                        break
                    prompt = _build_r2_prompt(obs)
                    examples.append({
                        "prompt": [
                            {"role": "system", "content": R2_SYSTEM_PROMPT},
                            {"role": "user",   "content": prompt},
                        ],
                    })
                    action = smart_fallback_r2(obs, assigned_set)
                    if action["action_type"] == "assign" and action.get("task_id"):
                        assigned_set.add(action["task_id"])
                    result = post(f"{ENV_BASE_URL}/project/step", {"action": action})
                    obs    = result.get("observation", obs)
                    if result.get("done", False):
                        break
            except Exception as e:
                print(f"  [WARN] R2 ep{ep} failed: {e}", flush=True)  # [FIX-T9]

    print(f"  [DATASET] Total examples: {len(examples)}", flush=True)
    if not examples:
        print("[ERROR] Dataset is empty — check server connectivity", flush=True)
        sys.exit(1)
    return Dataset.from_list(examples)


# ── SFT dataset builder ────────────────────────────────────────────────────────
# [FIX-T3] Used for SFT warm-up phase. Adds 'completion' field (the rule-based action).

def build_sft_dataset(n_examples: int = 100, phase: str = "both"):
    """
    Build a supervised fine-tuning dataset with (prompt, completion) pairs.
    The completion is the rule-based action — good enough to teach JSON format.
    """
    try:
        from datasets import Dataset
    except ImportError:
        sys.exit(1)

    import requests

    def post(url: str, payload: dict) -> dict:
        r = requests.post(url, json=payload, timeout=60)
        r.raise_for_status()
        return r.json()

    examples   = []
    tasks_r1   = R1_TASKS if phase in ("r1", "both") else []
    tasks_r2   = R2_TASKS if phase in ("r2", "both") else []
    per_task   = max(1, n_examples // max(1, len(tasks_r1) + len(tasks_r2)))
    SAMPLE_PER = 4

    for task_name in tasks_r1:
        for ep in range(per_task):
            try:
                obs = post(f"{ENV_BASE_URL}/reset", {"task_name": task_name, "seed": ep + 1000})
                for _ in range(SAMPLE_PER):
                    if obs.get("done", False):
                        break
                    action  = smart_fallback_r1(obs)
                    prompt  = _build_r1_prompt(obs)
                    completion = json.dumps(action)
                    examples.append({
                        "prompt": [
                            {"role": "system", "content": R1_SYSTEM_PROMPT},
                            {"role": "user",   "content": prompt},
                        ],
                        "completion": completion,
                    })
                    result = post(f"{ENV_BASE_URL}/step", {"action": action})
                    obs = result.get("observation", obs)
                    if result.get("done", False):
                        break
            except Exception as e:
                print(f"  [WARN] SFT R1 ep{ep} failed: {e}", flush=True)

    for task_name in tasks_r2:
        for ep in range(per_task):
            try:
                obs = post(f"{ENV_BASE_URL}/project/reset",
                           {"task_name": task_name, "seed": ep + 1000})
                assigned: set[str] = set()
                for _ in range(SAMPLE_PER):
                    if obs.get("done", False):
                        break
                    action = smart_fallback_r2(obs, assigned)
                    if action["action_type"] == "assign" and action.get("task_id"):
                        assigned.add(action["task_id"])
                    prompt     = _build_r2_prompt(obs)
                    completion = json.dumps(action)
                    examples.append({
                        "prompt": [
                            {"role": "system", "content": R2_SYSTEM_PROMPT},
                            {"role": "user",   "content": prompt},
                        ],
                        "completion": completion,
                    })
                    result = post(f"{ENV_BASE_URL}/project/step", {"action": action})
                    obs = result.get("observation", obs)
                    if result.get("done", False):
                        break
            except Exception as e:
                print(f"  [WARN] SFT R2 ep{ep} failed: {e}", flush=True)

    print(f"  [SFT DATASET] Total examples: {len(examples)}", flush=True)
    return Dataset.from_list(examples)


# ── Model loader ───────────────────────────────────────────────────────────────

def load_model_and_tokenizer(model_name: str):
    """
    Load model with Unsloth 4-bit QLoRA. Falls back to HF+PEFT if unavailable.
    [FIX-T5] Sets pad_token = eos_token for Llama/Qwen (they have no pad token by default).
    """
    try:
        from unsloth import FastLanguageModel
        print(f"[INFO] Loading {model_name} with Unsloth 4-bit QLoRA...", flush=True)
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_name,
            max_seq_length=2048,
            dtype=None,
            load_in_4bit=True,
            token=HF_TOKEN or None,
        )
        model = FastLanguageModel.get_peft_model(
            model,
            r=16,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj"],
            lora_alpha=32,
            lora_dropout=0.05,
            bias="none",
            use_gradient_checkpointing="unsloth",
            random_state=42,
        )
        # [FIX-T5]
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"[INFO] Unsloth loaded. Trainable params: {n_trainable:,}", flush=True)
        return model, tokenizer, "unsloth"
    except ImportError:
        print("[WARN] Unsloth not available. Falling back to HF + PEFT.", flush=True)
        return _load_hf_model(model_name)


def _load_hf_model(model_name: str):
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        from peft import get_peft_model, LoraConfig, TaskType
        import torch

        bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        tokenizer = AutoTokenizer.from_pretrained(model_name, token=HF_TOKEN or None)
        # [FIX-T5]
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            model_name, quantization_config=bnb, device_map="auto",
            token=HF_TOKEN or None,
        )
        lora_cfg = LoraConfig(
            r=16, lora_alpha=32, lora_dropout=0.05,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj"],
            task_type=TaskType.CAUSAL_LM,
        )
        model = get_peft_model(model, lora_cfg)
        print("[INFO] HF+PEFT model loaded.", flush=True)
        return model, tokenizer, "hf"
    except Exception as e:
        print(f"[ERROR] Cannot load model: {e}", flush=True)
        sys.exit(1)


# ── SFT trainer ────────────────────────────────────────────────────────────────

def run_sft(model, tokenizer, phase: str, n_examples: int, output_dir: str):
    """
    [FIX-T3] SFT warm-up: teach the model JSON format before GRPO exploration.
    Uses TRL SFTTrainer with the rule-based (obs, action) pairs.
    """
    print(f"\n[SFT] Warm-up phase ({n_examples} examples)...", flush=True)
    try:
        from trl import SFTTrainer, SFTConfig
    except ImportError:
        print("[WARN] SFTTrainer not in this trl version — skipping SFT warm-up", flush=True)
        return model

    sft_data = build_sft_dataset(n_examples=n_examples, phase=phase)

    def format_fn(example):
        """Convert chat messages + completion to a single formatted string."""
        parts = []
        for msg in example["prompt"]:
            parts.append(f"<|{msg['role']}|>\n{msg['content']}")
        parts.append(f"<|assistant|>\n{example['completion']}")
        return {"text": "\n".join(parts)}

    sft_data = sft_data.map(format_fn)

    sft_dir  = str(Path(output_dir) / "sft_warmup")
    sft_conf = SFTConfig(
        output_dir=sft_dir,
        dataset_text_field="text",
        max_seq_length=1024,
        report_to="none",
        **SFT_CONFIG,
    )
    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=sft_data,
        args=sft_conf,
    )
    trainer.train()
    print("[SFT] Warm-up complete.", flush=True)
    return trainer.model


# ── GRPO trainer ───────────────────────────────────────────────────────────────

def train(
    phase: str         = "both",
    n_dataset_examples: int = 200,
    output_dir: str    = "results/trained_model",
    push_to_hub: bool  = False,
    sft_epochs: int    = 0,     # 0 = skip SFT warm-up
    gpu_tier: str      = "t4",  # "t4", "a10g", or "a100"
):
    print(f"\n{'='*60}", flush=True)
    print(f" GRPO TRAINING — Phase: {phase.upper()} | GPU: {gpu_tier.upper()}", flush=True)
    print(f" Model:  {MODEL_NAME}", flush=True)
    print(f" Server: {ENV_BASE_URL}", flush=True)
    print(f" SFT warm-up epochs: {sft_epochs}", flush=True)
    print(f"{'='*60}\n", flush=True)

    # Adjust config for GPU tier
    cfg = dict(GRPO_CONFIG)
    if gpu_tier == "a10g":
        cfg["per_device_train_batch_size"] = 2
        cfg["num_generations"]             = 4
    elif gpu_tier == "a100":
        cfg["per_device_train_batch_size"] = 4
        cfg["num_generations"]             = 4
        cfg["gradient_accumulation_steps"] = 4

    # 1. Load model
    model, tokenizer, backend = load_model_and_tokenizer(MODEL_NAME)

    # 2. SFT warm-up (optional)
    if sft_epochs > 0:
        sft_n = max(50, n_dataset_examples // 3)
        SFT_CONFIG["num_train_epochs"] = sft_epochs
        model = run_sft(model, tokenizer, phase, sft_n, output_dir)

    # 3. Build GRPO dataset
    print("[INFO] Building GRPO training dataset...", flush=True)
    dataset = build_grpo_dataset(n_examples=n_dataset_examples, phase=phase)

    # 4. Build reward function
    reward_fn = make_reward_fn(ENV_BASE_URL, phase)

    # 5. Configure GRPOTrainer
    try:
        from trl import GRPOConfig, GRPOTrainer
    except ImportError:
        print("[ERROR] trl not installed. Run: pip install trl>=0.9.0", flush=True)
        sys.exit(1)

    # [FIX-T6] graceful version check for older trl
    import trl as _trl
    _trl_version = tuple(int(x) for x in _trl.__version__.split(".")[:2])
    if _trl_version < (0, 9):
        print(f"[WARN] trl {_trl.__version__} detected — recommend trl>=0.9.0", flush=True)
        # Remove keys that don't exist in older versions
        for key in ("warmup_ratio",):
            cfg.pop(key, None)

    grpo_config = GRPOConfig(
        output_dir=output_dir,
        report_to="none",
        **cfg,
    )

    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=reward_fn,
        args=grpo_config,
        train_dataset=dataset,
    )

    # 6. Train
    print("[INFO] Starting GRPO training...", flush=True)
    t0 = time.time()
    trainer.train()
    elapsed = time.time() - t0
    print(f"\n[INFO] Training complete in {elapsed/60:.1f} min", flush=True)

    # 7. Save
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"[INFO] Model saved to {output_dir}", flush=True)

    # 8. Push to Hub
    if push_to_hub and HF_REPO_ID:
        print(f"[INFO] Pushing to HF Hub: {HF_REPO_ID}", flush=True)
        # [FIX-T7] Merge LoRA weights before push so hub model is self-contained
        if backend == "unsloth":
            try:
                from unsloth import FastLanguageModel
                merged = model.merge_and_unload()
                merged.push_to_hub(HF_REPO_ID, token=HF_TOKEN)
                tokenizer.push_to_hub(HF_REPO_ID, token=HF_TOKEN)
                print(f"[INFO] Merged model pushed to https://huggingface.co/{HF_REPO_ID}", flush=True)
            except Exception as e:
                print(f"[WARN] Merge failed ({e}), pushing LoRA adapter only", flush=True)
                model.push_to_hub(HF_REPO_ID, token=HF_TOKEN)
                tokenizer.push_to_hub(HF_REPO_ID, token=HF_TOKEN)
        else:
            model.push_to_hub(HF_REPO_ID, token=HF_TOKEN)
            tokenizer.push_to_hub(HF_REPO_ID, token=HF_TOKEN)

    return output_dir


# ── Smoke test ─────────────────────────────────────────────────────────────────

def smoke_test():
    """
    Smoke test — no GPU, no model loading.
    Verifies: server reachability, R1/R2 env steps, SFT+GRPO dataset pipeline.
    """
    print("\n=== SMOKE TEST (rule-based, no GPU) ===\n", flush=True)
    import requests as _req

    # Health checks
    try:
        r1h = _req.get(f"{ENV_BASE_URL}/health",         timeout=10).json()
        r2h = _req.get(f"{ENV_BASE_URL}/project/health", timeout=10).json()
        print(f"[OK] R1 health: {r1h}", flush=True)
        print(f"[OK] R2 health: {r2h}", flush=True)
    except Exception as e:
        print(f"[ERROR] Server not reachable: {e}", flush=True)
        sys.exit(1)

    results = {}

    # ── R1 test ──────────────────────────────────────────────────────────────
    task = "easy_sprint"
    print(f"\n[R1] {task} (10 steps)...", flush=True)
    try:
        obs = _req.post(f"{ENV_BASE_URL}/reset",
                        json={"task_name": task, "seed": 42}, timeout=30).json()
        total_r = 0.0
        for i in range(10):
            if obs.get("done", False):
                break
            action = smart_fallback_r1(obs)
            result = _req.post(f"{ENV_BASE_URL}/step",
                                json={"action": action}, timeout=30).json()
            obs     = result.get("observation", obs)
            total_r += result.get("reward", 0.0)
            print(f"  step {i+1}: {action['action_type']} "
                  f"day={obs.get('current_day','?')} "
                  f"done={obs.get('tasks_completed',0)} "
                  f"reward={result.get('reward',0):.3f}", flush=True)
            if result.get("done", False):
                break
        results["r1/easy_sprint"] = round(total_r, 3)
        print(f"[OK] R1 cumulative reward: {total_r:.3f}", flush=True)
    except Exception as e:
        print(f"  [ERROR] {e}", flush=True)
        results["r1/easy_sprint"] = None

    # ── R2 test ──────────────────────────────────────────────────────────────
    task = "project_easy"
    print(f"\n[R2] {task} (8 steps)...", flush=True)
    try:
        obs      = _req.post(f"{ENV_BASE_URL}/project/reset",
                             json={"task_name": task, "seed": 42}, timeout=30).json()
        assigned: set[str] = set()
        for i in range(8):
            if obs.get("done", False):
                break
            action = smart_fallback_r2(obs, assigned)
            if action["action_type"] == "assign" and action.get("task_id"):
                assigned.add(action["task_id"])
            result = _req.post(f"{ENV_BASE_URL}/project/step",
                                json={"action": action}, timeout=30).json()
            obs = result.get("observation", obs)
            print(f"  step {i+1}: {action['action_type']} "
                  f"task={action.get('task_id')} "
                  f"day={obs.get('current_day','?')} "
                  f"sprint={obs.get('current_sprint','?')} "
                  f"reward={result.get('reward',0):.3f} "
                  f"inst={obs.get('instruction_following_score',0):.2f}", flush=True)
            if result.get("done", False):
                break
        results["r2/project_easy"] = round(obs.get("cumulative_reward", 0), 3)
        print(f"[OK] R2 cumulative reward: {obs.get('cumulative_reward',0):.3f}", flush=True)
    except Exception as e:
        print(f"  [ERROR] {e}", flush=True)
        results["r2/project_easy"] = None

    # ── Dataset pipeline test ─────────────────────────────────────────────────
    print(f"\n[DATASET] Testing GRPO dataset (12 examples)...", flush=True)
    try:
        ds = build_grpo_dataset(n_examples=12, phase="both")
        print(f"  [OK] GRPO dataset size: {len(ds)}", flush=True)
        print(f"  [OK] Keys: {list(ds[0].keys())}", flush=True)
    except Exception as e:
        print(f"  [WARN] GRPO dataset failed: {e}", flush=True)

    print(f"\n[DATASET] Testing SFT dataset (12 examples)...", flush=True)
    try:
        sft_ds = build_sft_dataset(n_examples=12, phase="both")
        print(f"  [OK] SFT dataset size: {len(sft_ds)}", flush=True)
        print(f"  [OK] Keys: {list(sft_ds[0].keys())}", flush=True)
    except Exception as e:
        print(f"  [WARN] SFT dataset failed: {e}", flush=True)

    print(f"\n=== SMOKE TEST RESULTS ===", flush=True)
    for k, v in results.items():
        status = "[OK]" if v is not None else "[FAIL]"
        print(f"  {status} {k}: {v}", flush=True)
    print(f"\n✅ Smoke test complete. Server is ready for GPU training.", flush=True)
    print(f"\n Recommended training command (A100):", flush=True)
    print(f"   python train_llm.py --phase both --episodes 300 "
          f"--sft-epochs 2 --gpu-tier a100 --output results/trained_model --push", flush=True)


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="SFT+GRPO training for AI Sprint Manager")
    parser.add_argument("--smoke-test",  action="store_true",
                        help="Run smoke test (no GPU). Do this locally first.")
    parser.add_argument("--phase",       choices=["r1", "r2", "both", "sft"], default="both",
                        help="Training phase. 'sft' runs SFT only.")
    parser.add_argument("--episodes",    type=int, default=200,
                        help="GRPO dataset examples to collect (default: 200)")
    parser.add_argument("--sft-epochs",  type=int, default=0,
                        help="SFT warm-up epochs before GRPO (default: 0 = skip)")
    parser.add_argument("--gpu-tier",    choices=["t4", "a10g", "a100"], default="t4",
                        help="GPU tier for batch size scaling (default: t4)")
    parser.add_argument("--output",      type=str, default="results/trained_model")
    parser.add_argument("--push",        action="store_true",
                        help="Push trained model to HF Hub (requires HF_REPO_ID)")
    args = parser.parse_args()

    if args.smoke_test:
        smoke_test()
        return

    if args.phase == "sft":
        # SFT only — useful to check format learning before GRPO
        model, tokenizer, _ = load_model_and_tokenizer(MODEL_NAME)
        run_sft(model, tokenizer, "both", 200, args.output)
        tokenizer.save_pretrained(args.output)
        return

    train(
        phase=args.phase,
        n_dataset_examples=args.episodes,
        output_dir=args.output,
        push_to_hub=args.push,
        sft_epochs=args.sft_epochs,
        gpu_tier=args.gpu_tier,
    )


if __name__ == "__main__":
    main()