"""
OpenEnv-compliant Pydantic models: Action, Observation, State.
"""
from __future__ import annotations
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field


# ─── Action ──────────────────────────────────────────────────────────────────

class SprintAction(BaseModel):
    """
    The agent's action each step.

    action_type options:
      - "assign"       : Assign a task to a developer
      - "reassign"     : Move task from one dev to another
      - "reprioritize" : Change a task's priority
      - "skip"         : Do nothing this step (let sprint progress)
      - "unblock"      : Mark a blocked task as actionable again

    task_id    : ID of the task to act on (e.g. "T1")
    dev_id     : ID of developer to assign to (e.g. "dev1")
    new_priority: New priority value 1-5 (for reprioritize)
    """
    action_type: str = Field(
        default="skip",
        description="One of: assign, reassign, reprioritize, skip, unblock"
    )
    task_id: Optional[str] = Field(default=None, description="Task ID to act on")
    dev_id: Optional[str] = Field(default=None, description="Developer ID to assign to")
    new_priority: Optional[int] = Field(default=None, description="New priority 1-5")


# ─── Observation ─────────────────────────────────────────────────────────────

class SprintObservation(BaseModel):
    """What the agent sees after each step."""
    # Sprint state
    current_day: int = Field(description="Current sprint day (1-10)")
    sprint_length: int = Field(description="Total sprint length in days")
    task_id: Optional[str] = Field(default=None, description="Active task context")

    # Team snapshot
    developers: List[Dict[str, Any]] = Field(description="List of developer states")
    tasks: List[Dict[str, Any]] = Field(description="All tasks and their statuses")

    # Reward signals
    reward: float = Field(default=0.0, description="Step reward")
    cumulative_reward: float = Field(default=0.0, description="Total reward so far")

    # Progress metrics
    tasks_completed: int = Field(default=0)
    tasks_missed: int = Field(default=0)
    tasks_in_progress: int = Field(default=0)
    tasks_backlog: int = Field(default=0)
    workload_balance_score: float = Field(default=0.0, description="0=unbalanced, 1=balanced")

    # Events that just happened
    events: List[str] = Field(default_factory=list, description="Events this step")

    # Terminal flag
    done: bool = Field(default=False)
    info: Dict[str, Any] = Field(default_factory=dict)


# ─── State ───────────────────────────────────────────────────────────────────

class SprintState(BaseModel):
    """Full internal state (for state() endpoint)."""
    episode_id: str
    task_name: str
    current_day: int
    sprint_length: int
    step_count: int
    tasks: List[Dict[str, Any]]
    developers: List[Dict[str, Any]]
    cumulative_reward: float
    done: bool
    events_log: List[str]