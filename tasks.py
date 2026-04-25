"""
Task and Developer data classes for the Sprint Manager environment.
"""
from __future__ import annotations
import random
from dataclasses import dataclass, field
from typing import Optional
from enum import Enum


class TaskType(str, Enum):
    FEATURE = "feature"
    BUG = "bug"
    URGENT_BUG = "urgent_bug"
    TECH_DEBT = "tech_debt"


class TaskStatus(str, Enum):
    BACKLOG = "backlog"
    IN_PROGRESS = "in_progress"
    BLOCKED = "blocked"
    DONE = "done"
    MISSED = "missed"


@dataclass
class Task:
    id: str
    name: str
    task_type: TaskType
    priority: int          # 1 (highest) to 5 (lowest)
    effort: int            # story points: 1-8
    deadline: int          # sprint day deadline (1-10)
    required_skill: str    # "frontend", "backend", "devops", "fullstack"
    status: TaskStatus = TaskStatus.BACKLOG
    assigned_to: Optional[str] = None
    progress: float = 0.0  # 0.0 to 1.0
    days_in_progress: int = 0
    created_day: int = 1

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "task_type": self.task_type.value,
            "priority": self.priority,
            "effort": self.effort,
            "deadline": self.deadline,
            "required_skill": self.required_skill,
            "status": self.status.value,
            "assigned_to": self.assigned_to,
            "progress": round(self.progress, 2),
            "days_in_progress": self.days_in_progress,
            "created_day": self.created_day,
        }


@dataclass
class Developer:
    id: str
    name: str
    skill: str             # "frontend", "backend", "devops", "fullstack"
    capacity: int          # max story points per day
    current_load: int = 0  # current assigned story points
    assigned_tasks: list = field(default_factory=list)
    is_available: bool = True
    productivity: float = 1.0  # multiplier: 0.5 to 1.5

    def can_take_task(self, task: Task) -> bool:
        """Check if dev can take this task."""
        if not self.is_available:
            return False
        if self.current_load >= self.capacity:
            return False
        skill_match = (
            self.skill == task.required_skill
            or self.skill == "fullstack"
            or task.required_skill == "fullstack"
        )
        return skill_match

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "skill": self.skill,
            "capacity": self.capacity,
            "current_load": self.current_load,
            "assigned_tasks": list(self.assigned_tasks),
            "is_available": self.is_available,
            "productivity": round(self.productivity, 2),
        }


def make_easy_sprint() -> tuple[list[Task], list[Developer]]:
    """Easy: 3 devs, 5 tasks, no surprises."""
    developers = [
        Developer("dev1", "Alice", "backend", capacity=5, productivity=1.2),
        Developer("dev2", "Bob", "frontend", capacity=4, productivity=1.0),
        Developer("dev3", "Carol", "fullstack", capacity=6, productivity=1.1),
    ]
    tasks = [
        Task("T1", "User login API", TaskType.FEATURE, 2, 3, 5, "backend"),
        Task("T2", "Dashboard UI", TaskType.FEATURE, 2, 3, 6, "frontend"),
        Task("T3", "Fix CSS bug", TaskType.BUG, 1, 1, 3, "frontend"),
        Task("T4", "Database migration", TaskType.FEATURE, 3, 4, 7, "backend"),
        Task("T5", "Setup CI/CD", TaskType.FEATURE, 3, 3, 8, "devops"),
    ]
    return tasks, developers


def make_medium_sprint() -> tuple[list[Task], list[Developer]]:
    """Medium: 4 devs, 8 tasks, mid-sprint bug + delay event."""
    developers = [
        Developer("dev1", "Alice", "backend", capacity=5, productivity=1.1),
        Developer("dev2", "Bob", "frontend", capacity=4, productivity=0.9),
        Developer("dev3", "Carol", "devops", capacity=5, productivity=1.0),
        Developer("dev4", "Dave", "fullstack", capacity=6, productivity=1.2),
    ]
    tasks = [
        Task("T1", "Auth service", TaskType.FEATURE, 1, 5, 5, "backend"),
        Task("T2", "Profile page UI", TaskType.FEATURE, 2, 4, 6, "frontend"),
        Task("T3", "Payment integration", TaskType.FEATURE, 1, 6, 7, "backend"),
        Task("T4", "Docker deployment", TaskType.FEATURE, 2, 3, 5, "devops"),
        Task("T5", "Login page", TaskType.FEATURE, 2, 3, 5, "frontend"),
        Task("T6", "Prod DB crash", TaskType.BUG, 1, 2, 3, "backend"),
        Task("T7", "API rate limiting", TaskType.TECH_DEBT, 4, 3, 9, "backend"),
        Task("T8", "Mobile layout fix", TaskType.BUG, 2, 2, 4, "frontend"),
    ]
    return tasks, developers


def make_hard_sprint() -> tuple[list[Task], list[Developer]]:
    """Hard: 5 devs, 12 tasks, random absences, cascading bugs, scope creep."""
    developers = [
        Developer("dev1", "Alice", "backend", capacity=5, productivity=1.3),
        Developer("dev2", "Bob", "frontend", capacity=4, productivity=0.8),
        Developer("dev3", "Carol", "devops", capacity=5, productivity=1.0),
        Developer("dev4", "Dave", "fullstack", capacity=7, productivity=1.1),
        Developer("dev5", "Eve", "backend", capacity=5, productivity=0.9),
    ]
    tasks = [
        Task("T1", "Microservices refactor", TaskType.FEATURE, 1, 8, 5, "backend"),
        Task("T2", "Real-time notifications", TaskType.FEATURE, 1, 6, 5, "fullstack"),
        Task("T3", "Security audit fixes", TaskType.BUG, 1, 5, 4, "backend"),
        Task("T4", "k8s migration", TaskType.FEATURE, 2, 7, 6, "devops"),
        Task("T5", "New homepage redesign", TaskType.FEATURE, 2, 5, 6, "frontend"),
        Task("T6", "Payment gateway bug", TaskType.URGENT_BUG, 1, 3, 3, "backend"),
        Task("T7", "Search performance", TaskType.TECH_DEBT, 3, 4, 7, "backend"),
        Task("T8", "A/B testing framework", TaskType.FEATURE, 3, 5, 8, "fullstack"),
        Task("T9", "CDN integration", TaskType.FEATURE, 3, 3, 7, "devops"),
        Task("T10", "OAuth2 provider", TaskType.FEATURE, 2, 6, 6, "backend"),
        Task("T11", "Analytics dashboard", TaskType.FEATURE, 3, 5, 8, "fullstack"),
        Task("T12", "Accessibility audit", TaskType.TECH_DEBT, 4, 3, 9, "frontend"),
    ]
    return tasks, developers