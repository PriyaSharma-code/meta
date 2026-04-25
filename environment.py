"""
Sprint Manager RL Environment — core logic.
"""
from __future__ import annotations
import random
import uuid
from typing import Optional, List, Tuple

from sprint_env.models import SprintAction, SprintObservation, SprintState
from sprint_env.tasks import (
    Task, Developer, TaskStatus, TaskType,
    make_easy_sprint, make_medium_sprint, make_hard_sprint,
)
from sprint_env.graders import grade_easy, grade_medium, grade_hard

SPRINT_LENGTH = 10  # days per episode
VALID_TASK_NAMES = ["easy_sprint", "medium_sprint", "hard_sprint"]


class SprintManagerEnv:
    """
    OpenEnv-compliant Sprint Manager environment.
    
    Episode lifecycle:
      reset(task_name) → day 1 of sprint
      step(action)     → agent assigns/reassigns tasks, sprint progresses 1 day
      state()          → current internal state dict
    """

    def __init__(self):
        self._episode_id: str = ""
        self._task_name: str = "easy_sprint"
        self._current_day: int = 1
        self._step_count: int = 0
        self._done: bool = False
        self._tasks: List[Task] = []
        self._developers: List[Developer] = []
        self._cumulative_reward: float = 0.0
        self._events_log: List[str] = []
        self._grader = grade_easy

    # ── Reset ────────────────────────────────────────────────────────────────

    def reset(
        self,
        task_name: str = "easy_sprint",
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
    ) -> SprintObservation:
        if seed is not None:
            random.seed(seed)

        self._episode_id = episode_id or str(uuid.uuid4())
        self._task_name = task_name if task_name in VALID_TASK_NAMES else "easy_sprint"
        self._current_day = 1
        self._step_count = 0
        self._done = False
        self._cumulative_reward = 0.0
        self._events_log = []

        if self._task_name == "easy_sprint":
            self._tasks, self._developers = make_easy_sprint()
            self._grader = grade_easy
        elif self._task_name == "medium_sprint":
            self._tasks, self._developers = make_medium_sprint()
            self._grader = grade_medium
        else:
            self._tasks, self._developers = make_hard_sprint()
            self._grader = grade_hard

        return self._make_observation(reward=0.0, events=["Sprint started!"])

    # ── Step ─────────────────────────────────────────────────────────────────

    def step(self, action: SprintAction) -> Tuple[SprintObservation, float, bool, dict]:
        if self._done:
            obs = self._make_observation(reward=0.0, events=["Episode already done."])
            return obs, 0.0, True, {}

        self._step_count += 1
        events: List[str] = []
        step_reward = 0.0

        # 1. Apply agent action
        action_reward, action_events = self._apply_action(action)
        step_reward += action_reward
        events.extend(action_events)

        # 2. Simulate one day of work
        day_reward, day_events = self._simulate_day()
        step_reward += day_reward
        events.extend(day_events)

        # 3. Inject random events (medium/hard only)
        if self._task_name != "easy_sprint":
            rand_reward, rand_events = self._random_events()
            step_reward += rand_reward
            events.extend(rand_events)

        # 4. Advance day
        self._current_day += 1

        # 5. Check termination
        all_resolved = all(
            t.status in (TaskStatus.DONE, TaskStatus.MISSED)
            for t in self._tasks
        )
        if self._current_day > SPRINT_LENGTH or all_resolved:
            self._done = True
            # Final scoring bonus
            final_score = self._grader(self._tasks, self._developers, SPRINT_LENGTH)
            final_bonus = final_score * 10.0
            step_reward += final_bonus
            events.append(f"Sprint ended! Final score: {final_score:.2f}")

        self._cumulative_reward += step_reward
        self._events_log.extend(events)

        obs = self._make_observation(reward=step_reward, events=events)
        return obs, step_reward, self._done, {"final_score": self._grader(self._tasks, self._developers, SPRINT_LENGTH)}

    # ── State ─────────────────────────────────────────────────────────────────

    @property
    def state(self) -> SprintState:
        return SprintState(
            episode_id=self._episode_id,
            task_name=self._task_name,
            current_day=self._current_day,
            sprint_length=SPRINT_LENGTH,
            step_count=self._step_count,
            tasks=[t.to_dict() for t in self._tasks],
            developers=[d.to_dict() for d in self._developers],
            cumulative_reward=round(self._cumulative_reward, 4),
            done=self._done,
            events_log=self._events_log[-20:],  # last 20 events
        )

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _apply_action(self, action: SprintAction) -> Tuple[float, List[str]]:
        reward = 0.0
        events = []
        atype = (action.action_type or "skip").lower()

        task = self._find_task(action.task_id)
        dev = self._find_dev(action.dev_id)

        if atype == "assign":
            if task is None:
                events.append(f"Invalid assign: task {action.task_id} not found.")
                reward -= 0.2
            elif dev is None:
                events.append(f"Invalid assign: dev {action.dev_id} not found.")
                reward -= 0.2
            elif task.status != TaskStatus.BACKLOG:
                events.append(f"Task {task.id} is not in backlog (status={task.status.value}).")
                reward -= 0.1
            elif not dev.can_take_task(task):
                events.append(f"Dev {dev.id} can't take task {task.id} (capacity/skill mismatch).")
                reward -= 0.15
            else:
                task.status = TaskStatus.IN_PROGRESS
                task.assigned_to = dev.id
                dev.assigned_tasks.append(task.id)
                dev.current_load += task.effort
                # Reward: higher for matching skill, urgent tasks, high priority
                skill_bonus = 0.3 if dev.skill == task.required_skill else 0.1
                priority_bonus = (6 - task.priority) * 0.1
                reward += 0.5 + skill_bonus + priority_bonus
                events.append(f"Assigned {task.id} ({task.name}) → {dev.name}")

        elif atype == "reassign":
            if task is None or dev is None:
                events.append("Invalid reassign: task or dev not found.")
                reward -= 0.2
            elif task.status not in (TaskStatus.IN_PROGRESS, TaskStatus.BLOCKED):
                events.append(f"Cannot reassign task {task.id} (not in progress).")
                reward -= 0.1
            else:
                # Remove from old dev
                old_dev = self._find_dev(task.assigned_to)
                if old_dev:
                    if task.id in old_dev.assigned_tasks:
                        old_dev.assigned_tasks.remove(task.id)
                    old_dev.current_load = max(0, old_dev.current_load - task.effort)
                # Assign to new dev
                task.assigned_to = dev.id
                task.status = TaskStatus.IN_PROGRESS
                dev.assigned_tasks.append(task.id)
                dev.current_load += task.effort
                reward += 0.2
                events.append(f"Reassigned {task.id} → {dev.name}")

        elif atype == "reprioritize":
            if task is None:
                events.append(f"Task {action.task_id} not found for reprioritize.")
                reward -= 0.1
            elif action.new_priority not in range(1, 6):
                events.append(f"Invalid priority {action.new_priority}.")
                reward -= 0.05
            else:
                old_p = task.priority
                task.priority = action.new_priority
                reward += 0.1
                events.append(f"Reprioritized {task.id}: {old_p} → {action.new_priority}")

        elif atype == "unblock":
            if task is None:
                events.append(f"Task {action.task_id} not found.")
                reward -= 0.1
            elif task.status != TaskStatus.BLOCKED:
                events.append(f"Task {task.id} is not blocked.")
            else:
                task.status = TaskStatus.IN_PROGRESS
                reward += 0.3
                events.append(f"Unblocked task {task.id}")

        elif atype == "skip":
            reward -= 0.05  # small penalty for inaction
            events.append("Agent chose to skip (no action).")

        else:
            events.append(f"Unknown action type: {atype}")
            reward -= 0.2

        return reward, events

    def _simulate_day(self) -> Tuple[float, List[str]]:
        """Advance each in-progress task by one day of work."""
        reward = 0.0
        events = []

        for task in self._tasks:
            if task.status != TaskStatus.IN_PROGRESS:
                continue

            dev = self._find_dev(task.assigned_to)
            if dev is None or not dev.is_available:
                continue

            # Progress = dev productivity / task effort per day
            daily_progress = (dev.productivity * dev.capacity) / (task.effort * SPRINT_LENGTH / 2)
            daily_progress = min(daily_progress, 0.5)  # cap at 50% per day
            task.progress = min(1.0, task.progress + daily_progress)
            task.days_in_progress += 1

            if task.progress >= 1.0:
                task.status = TaskStatus.DONE
                task.progress = 1.0
                on_time = self._current_day <= task.deadline
                if on_time:
                    reward += (6 - task.priority) * 0.5 + 0.5
                    events.append(f"✅ {task.name} completed on time!")
                else:
                    reward += 0.1
                    events.append(f"⚠️ {task.name} completed LATE.")
                dev.assigned_tasks = [tid for tid in dev.assigned_tasks if tid != task.id]
                dev.current_load = max(0, dev.current_load - task.effort)

            # Check missed deadline
            elif self._current_day > task.deadline and task.status == TaskStatus.IN_PROGRESS:
                task.status = TaskStatus.MISSED
                reward -= (6 - task.priority) * 0.4
                events.append(f"❌ {task.name} MISSED deadline!")
                dev_obj = self._find_dev(task.assigned_to)
                if dev_obj:
                    if task.id in dev_obj.assigned_tasks:
                        dev_obj.assigned_tasks.remove(task.id)
                    dev_obj.current_load = max(0, dev_obj.current_load - task.effort)

        # Backlog tasks past deadline become missed
        for task in self._tasks:
            if task.status == TaskStatus.BACKLOG and self._current_day > task.deadline:
                task.status = TaskStatus.MISSED
                reward -= (6 - task.priority) * 0.3
                events.append(f"❌ {task.name} expired in backlog!")

        return reward, events

    def _random_events(self) -> Tuple[float, List[str]]:
        """Inject random events for medium/hard sprints."""
        reward = 0.0
        events = []

        # 15% chance of a dev going unavailable for a day
        for dev in self._developers:
            if dev.is_available and random.random() < 0.08:
                dev.is_available = False
                events.append(f"🤒 {dev.name} is unavailable today!")
            elif not dev.is_available:
                dev.is_available = True  # recovered
                events.append(f"💪 {dev.name} is back!")

        # Hard sprint: 20% chance of new urgent bug
        if self._task_name == "hard_sprint" and random.random() < 0.15:
            day = self._current_day
            bug_id = f"BUG_{day}"
            if not any(t.id == bug_id for t in self._tasks):
                new_bug = Task(
                    id=bug_id,
                    name=f"Urgent prod bug (day {day})",
                    task_type=TaskType.URGENT_BUG,
                    priority=1,
                    effort=2,
                    deadline=min(day + 2, SPRINT_LENGTH),
                    required_skill="backend",
                    created_day=day,
                )
                self._tasks.append(new_bug)
                reward -= 0.3
                events.append(f"🚨 New urgent bug appeared: {bug_id}!")

        return reward, events

    def _make_observation(self, reward: float, events: List[str]) -> SprintObservation:
        done_count = sum(1 for t in self._tasks if t.status == TaskStatus.DONE)
        missed_count = sum(1 for t in self._tasks if t.status == TaskStatus.MISSED)
        in_prog_count = sum(1 for t in self._tasks if t.status == TaskStatus.IN_PROGRESS)
        backlog_count = sum(1 for t in self._tasks if t.status == TaskStatus.BACKLOG)

        # Workload balance: 1 = perfectly balanced, 0 = unbalanced
        load_ratios = [
            d.current_load / d.capacity for d in self._developers if d.capacity > 0
        ]
        if load_ratios:
            mean = sum(load_ratios) / len(load_ratios)
            variance = sum((r - mean) ** 2 for r in load_ratios) / len(load_ratios)
            balance = max(0.0, 1.0 - variance ** 0.5)
        else:
            balance = 1.0

        return SprintObservation(
            current_day=self._current_day,
            sprint_length=SPRINT_LENGTH,
            task_id=self._task_name,
            developers=[d.to_dict() for d in self._developers],
            tasks=[t.to_dict() for t in self._tasks],
            reward=round(reward, 4),
            cumulative_reward=round(self._cumulative_reward + reward, 4),
            tasks_completed=done_count,
            tasks_missed=missed_count,
            tasks_in_progress=in_prog_count,
            tasks_backlog=backlog_count,
            workload_balance_score=round(balance, 4),
            events=events,
            done=self._done,
            info={
                "episode_id": self._episode_id,
                "step": self._step_count,
            },
        )

    def _find_task(self, task_id: Optional[str]) -> Optional[Task]:
        if not task_id:
            return None
        return next((t for t in self._tasks if t.id == task_id), None)

    def _find_dev(self, dev_id: Optional[str]) -> Optional[Developer]:
        if not dev_id:
            return None
        return next((d for d in self._developers if d.id == dev_id), None)