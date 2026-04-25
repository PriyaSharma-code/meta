---
title: AI Sprint Manager
emoji: 🤖
colorFrom: blue
colorTo: purple
sdk: docker
pinned: false
tags: [openenv]
---

# 🤖 AI Sprint Manager — OpenEnv

An RL environment where an AI agent acts as a **Tech Lead** managing agile sprints — assigning tasks to developers, balancing workloads, and responding to unexpected events.

## Why This Exists

Software teams lose countless hours to poor sprint planning. This environment enables RL agents to learn optimal task allocation strategies in a realistic simulation of real engineering team dynamics.

## Action Space

| Field | Type | Description |
|---|---|---|
| `action_type` | string | `assign`, `reassign`, `reprioritize`, `unblock`, `skip` |
| `task_id` | string | ID of task to act on (e.g. `"T1"`) |
| `dev_id` | string | ID of developer (e.g. `"dev1"`) |
| `new_priority` | int 1-5 | New priority for reprioritize action |

## Tasks

| ID | Difficulty | Description |
|---|---|---|
| `easy_sprint` | Easy | 3 devs, 5 tasks, no surprises |
| `medium_sprint` | Medium | 4 devs, 8 tasks, bugs + dev absences |
| `hard_sprint` | Hard | 5 devs, 12 tasks, urgent bugs mid-sprint |

## Setup
```bash
docker build -t ai-sprint-manager .
docker run -p 7860:7860 -p 8000:8000 ai-sprint-manager
```

## API
```bash
POST /reset   — start new episode
POST /step    — take an action  
GET  /state   — get current state
GET  /health  — health check
```