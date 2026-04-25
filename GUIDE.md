# 📖 Technical Guide — AI Sprint Manager OpenEnv

**What exactly are we building, how does it work, and how do you know it's working?**

---

## 🤔 What Are We Building — In Simple Words

Imagine you're a Tech Lead at a software company. Every 2 weeks (a "sprint"), your team gets a list of tasks — new features, bug fixes, etc. Your job is to decide:
- Which developer gets which task?
- What's most urgent?
- What do you do when a developer is sick or a new bug appears?

**We built a simulation of this scenario** so that an AI agent can practice making these decisions and learn to get better over time — the same way a human learns from experience.

The AI agent plays the role of the Tech Lead. It looks at the current state of the sprint (who's working on what, what's due soon, who's available) and decides what action to take next.

---

## 🏗️ How The System Is Built — Layer by Layer

```
┌─────────────────────────────────────────┐
│         YOU / AI AGENT                  │
│   (makes decisions: assign, skip, etc.) │
└──────────────────┬──────────────────────┘
                   │  HTTP requests
                   ▼
┌─────────────────────────────────────────┐
│         FASTAPI SERVER                  │
│   /reset  /step  /state  /health        │
│   (receives actions, returns results)   │
└──────────────────┬──────────────────────┘
                   │  calls
                   ▼
┌─────────────────────────────────────────┐
│      SPRINT ENVIRONMENT (core logic)    │
│   - Tracks tasks, developers, days      │
│   - Calculates rewards                  │
│   - Simulates work progress             │
│   - Fires random events (bugs, absence) │
└─────────────────────────────────────────┘
                   │  visualized by
                   ▼
┌─────────────────────────────────────────┐
│         GRADIO UI                       │
│   (Sprint board, dev workload, metrics) │
└─────────────────────────────────────────┘
```

---

## 🔄 What Happens Each "Step"

One step = one day in the sprint. Here's exactly what happens:

1. **Agent sees the current state** — all tasks, all developers, current day, recent events
2. **Agent picks an action** — e.g. "Assign Task T1 to Developer Alice"
3. **Environment processes the action** — checks if it's valid, applies it
4. **Sprint advances one day** — developers work on their assigned tasks, progress increases
5. **Random events may fire** — developer goes sick, new urgent bug appears
6. **Reward is calculated** — positive for good moves, negative for bad ones
7. **Agent sees the new state** — repeat until sprint ends (day 10)

---

## 🎯 The Three Tasks Explained

| Task | What Makes It Hard |
|------|-------------------|
| `easy_sprint` | 3 devs, 5 tasks, no surprises. Just assign correctly. |
| `medium_sprint` | 4 devs, 8 tasks. Devs randomly go unavailable. Bugs appear in backlog. |
| `hard_sprint` | 5 devs, 12 tasks. Urgent bugs appear mid-sprint. Dev absences. Tasks cascade. |

---

## 💰 How Rewards Work

The agent gets **positive rewards** for good behaviour:
- ✅ Assigning a task to the right skill dev: **+0.5 to +1.0**
- ✅ Task completed on time: **+0.5 to +2.5** (depends on priority)
- ✅ Unblocking a blocked task: **+0.3**

The agent gets **negative rewards** for bad behaviour:
- ❌ Assigning to wrong skill: **-0.15**
- ❌ Task missed deadline: **-0.3 to -1.5**
- ❌ Doing nothing (skip): **-0.05**
- ❌ Urgent bug missed: **-0.25 extra**

At the end of the sprint, a **final score (0.0–1.0)** is computed by the grader and a bonus reward is given.

---

## 📊 What the Scores Mean

```
easy_sprint:   0.0  → Agent assigned wrong skills all game
medium_sprint: 0.46 → Agent got some tasks done, missed others
hard_sprint:   0.0  → Cascade failures overwhelmed the agent
```

**A score of 0.0 doesn't mean broken** — it means the task is hard and the baseline LLM isn't smart enough yet. The environment is working correctly; the agent just needs improvement.

**A perfect agent would score:**
- easy: ~0.85–1.0
- medium: ~0.60–0.80
- hard: ~0.40–0.60

---

## 🧪 How To Test Everything Is Working Correctly

### Test 1: Health Check ✅
```bash
# Mac
curl http://localhost:7860/health

# Windows
Invoke-WebRequest -Uri http://localhost:7860/health -Method GET
```
**Expected:** `{"status":"ok","env":"ai-sprint-manager"}`
**What it means:** The server is running and reachable.

---

### Test 2: Reset Works ✅
```bash
# Mac
curl -X POST http://localhost:7860/reset \
  -H "Content-Type: application/json" \
  -d '{"task_name": "easy_sprint", "seed": 42}'
```
**Expected:** JSON with `current_day: 1`, 5 tasks all in `backlog` status, 3 developers.
**What it means:** A new sprint episode starts cleanly each time.

---

### Test 3: Step Works ✅
```bash
# Mac
curl -X POST http://localhost:7860/step \
  -H "Content-Type: application/json" \
  -d '{"action": {"action_type": "assign", "task_id": "T1", "dev_id": "dev1"}}'
```
**Expected:** `reward` around `+1.2`, task T1 now shows `status: in_progress`, assigned to `dev1`.
**What it means:** Actions are processed and the environment advances correctly.

---

### Test 4: Skill Mismatch is Rejected ✅
Try assigning a backend task to a frontend dev:
```bash
# T1 is backend, dev2 is frontend — should fail
curl -X POST http://localhost:7860/step \
  -H "Content-Type: application/json" \
  -d '{"action": {"action_type": "assign", "task_id": "T1", "dev_id": "dev2"}}'
```
**Expected:** `reward: -0.15`, event says "can't take task (capacity/skill mismatch)"
**What it means:** The environment correctly enforces skill constraints.

---

### Test 5: Sprint Ends After Day 10 ✅
Keep calling `/step` with skip actions:
```bash
curl -X POST http://localhost:7860/step \
  -H "Content-Type: application/json" \
  -d '{"action": {"action_type": "skip"}}'
```
After 10 calls, `done` should become `true`.
**What it means:** Episode boundaries work correctly.

---

### Test 6: State is Consistent ✅
```bash
curl http://localhost:7860/state
```
**Expected:** Full state including episode_id, current_day, all tasks and developers.
**What it means:** The state endpoint returns accurate internal state.

---

### Test 7: OpenEnv Validation ✅
```bash
openenv validate
```
**Expected:** `[OK] ai-sprint-manager: Ready for multi-mode deployment`
**What it means:** Your environment meets the official OpenEnv spec requirements.

---

### Test 8: Inference Script ✅
```bash
python inference.py
```
**Expected:** Scores printed for all 3 tasks, runtime under 20 minutes, no crashes.
**What it means:** The baseline agent can run against your environment end-to-end.

---

### Test 9: Docker Works ✅
```bash
docker build -t ai-sprint-manager .
docker run -p 7860:7860 ai-sprint-manager
# Then test health endpoint
curl http://localhost:7860/health
```
**What it means:** The environment can run in a clean isolated container.

---

### Test 10: UI Works ✅
Open http://localhost:7860 and:
1. Reset with `easy_sprint` → see 5 tasks in backlog
2. Assign T3 → dev2 → see reward and task move to in_progress
3. Skip a few times → see days advance
4. Let sprint end → see "SPRINT COMPLETE" message

**What it means:** The Gradio UI correctly communicates with the environment.

---

## 🚦 Quick Sanity Check Table

| Check | Command | Pass Condition |
|-------|---------|---------------|
| Server running | GET /health | `{"status":"ok"}` |
| Reset works | POST /reset | `current_day: 1`, tasks in backlog |
| Step works | POST /step | reward changes, day advances |
| Skill rules work | Assign wrong skill | Negative reward, error message |
| Sprint ends | 10 skip steps | `done: true` |
| Grader works | Check final score | Score between 0.0 and 1.0 |
| OpenEnv valid | `openenv validate` | `[OK]` message |
| Docker works | docker run + health | `{"status":"ok"}` |
| Inference runs | `python inference.py` | 3 scores printed, no crash |

---

## 🐛 Common Issues & Fixes

| Problem | Likely Cause | Fix |
|---------|-------------|-----|
| `ModuleNotFoundError` | Missing package | `pip install -r requirements.txt` |
| `Connection refused` | Server not running | Start `python ui.py` first |
| `401 Unauthorized` | Bad HF token | Check `.env` file, regenerate token |
| `Skill mismatch` error | Wrong dev for task | Check dev skills in observation |
| Score always 0.0 | All tasks missed | Assign tasks earlier in the sprint |
| Docker timeout | Slow internet | Run `docker pull python:3.11-slim` first |

---

## 📁 File Reference

| File | What It Does |
|------|-------------|
| `sprint_env/tasks.py` | Defines Task and Developer data, sprint scenarios |
| `sprint_env/models.py` | Pydantic models for Action, Observation, State |
| `sprint_env/environment.py` | Core RL logic: reset, step, reward calculation |
| `sprint_env/graders.py` | Scoring functions for easy/medium/hard |
| `server/app.py` | FastAPI endpoints (OpenEnv spec compliant) |
| `ui.py` | Gradio UI + combined app entry point |
| `inference.py` | Baseline LLM agent script |
| `openenv.yaml` | OpenEnv metadata and task definitions |
| `Dockerfile` | Container build instructions |
| `pyproject.toml` | Python project dependencies for OpenEnv |
| `.env` | Your secret tokens (never commit!) |