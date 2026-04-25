"""
Task graders: deterministic scoring functions for each difficulty level.
Score range: 0.0 (complete failure) to 1.0 (perfect).
"""
from __future__ import annotations
from typing import List
from sprint_env.tasks import Task, TaskStatus, TaskType


def grade_easy(tasks: List[Task], developers: list, sprint_length: int) -> float:
    """
    Easy grader: % of tasks completed on time, small penalties for overload.
    Score 0.0-1.0.
    """
    if not tasks:
        return 0.0

    completed_on_time = sum(
        1 for t in tasks
        if t.status == TaskStatus.DONE and t.days_in_progress <= (t.deadline - t.created_day + 1)
    )
    missed = sum(1 for t in tasks if t.status == TaskStatus.MISSED)
    total = len(tasks)

    completion_score = completed_on_time / total

    # Penalty: overloaded devs (any dev that was over capacity)
    overload_penalty = 0.0
    for dev in developers:
        if dev.current_load > dev.capacity:
            overload_penalty += 0.05

    score = max(0.0, min(1.0, completion_score - overload_penalty))
    return round(score, 4)


def grade_medium(tasks: List[Task], developers: list, sprint_length: int) -> float:
    """
    Medium grader: weights task priority — urgent/high-priority tasks matter more.
    Also considers workload balance across the team.
    """
    if not tasks:
        return 0.0

    priority_weights = {1: 3.0, 2: 2.0, 3: 1.5, 4: 1.0, 5: 0.5}
    total_weight = sum(priority_weights.get(t.priority, 1.0) for t in tasks)

    earned_weight = 0.0
    for t in tasks:
        w = priority_weights.get(t.priority, 1.0)
        if t.status == TaskStatus.DONE:
            # On time = full weight, late = 50%
            on_time = t.days_in_progress <= (t.deadline - t.created_day + 1)
            earned_weight += w if on_time else w * 0.5
        elif t.status == TaskStatus.IN_PROGRESS:
            earned_weight += w * t.progress * 0.3  # partial credit

    completion_score = earned_weight / total_weight if total_weight > 0 else 0.0

    # Workload balance: std deviation of load ratios
    load_ratios = [
        (dev.current_load / dev.capacity) for dev in developers if dev.capacity > 0
    ]
    if len(load_ratios) > 1:
        mean_ratio = sum(load_ratios) / len(load_ratios)
        variance = sum((r - mean_ratio) ** 2 for r in load_ratios) / len(load_ratios)
        std_dev = variance ** 0.5
        balance_score = max(0.0, 1.0 - std_dev)
    else:
        balance_score = 1.0

    score = 0.75 * completion_score + 0.25 * balance_score
    return round(max(0.0, min(1.0, score)), 4)


def grade_hard(tasks: List[Task], developers: list, sprint_length: int) -> float:
    """
    Hard grader: adds urgent bug penalty, dev burnout penalty, and streak bonuses.
    Requires near-perfect execution to score above 0.7.
    """
    if not tasks:
        return 0.0

    # Base: priority-weighted completion (same as medium)
    priority_weights = {1: 4.0, 2: 2.5, 3: 1.5, 4: 0.8, 5: 0.3}
    total_weight = sum(priority_weights.get(t.priority, 1.0) for t in tasks)
    earned_weight = 0.0

    urgent_bug_missed = False
    for t in tasks:
        w = priority_weights.get(t.priority, 1.0)
        if t.status == TaskStatus.DONE:
            on_time = t.days_in_progress <= (t.deadline - t.created_day + 1)
            earned_weight += w if on_time else w * 0.4
        elif t.status == TaskStatus.IN_PROGRESS:
            earned_weight += w * t.progress * 0.2
        elif t.status == TaskStatus.MISSED:
            if t.task_type == TaskType.URGENT_BUG:
                urgent_bug_missed = True

    completion_score = earned_weight / total_weight if total_weight > 0 else 0.0

    # Burnout penalty: devs working over 120% capacity
    burnout_penalty = sum(
        0.08 for dev in developers
        if dev.current_load > dev.capacity * 1.2
    )

    # Urgent bug penalty
    urgent_penalty = 0.25 if urgent_bug_missed else 0.0

    # Balance
    load_ratios = [
        (dev.current_load / dev.capacity) for dev in developers if dev.capacity > 0
    ]
    if len(load_ratios) > 1:
        mean_ratio = sum(load_ratios) / len(load_ratios)
        variance = sum((r - mean_ratio) ** 2 for r in load_ratios) / len(load_ratios)
        balance_score = max(0.0, 1.0 - variance ** 0.5)
    else:
        balance_score = 1.0

    score = (
        0.65 * completion_score
        + 0.20 * balance_score
        - burnout_penalty
        - urgent_penalty
    )
    return round(max(0.0, min(1.0, score)), 4)