# configs/task_config.py

from dataclasses import dataclass

@dataclass(frozen=True)
class Task:
    object_color: str
    target_side: str
    instruction: str

TASKS = [
    Task(
        object_color="red",
        target_side="left",
        instruction="Pick up the red cube and place it on the left.",
    ),
    Task(
        object_color="red",
        target_side="right",
        instruction="Pick up the red cube and place it on the right.",
    ),
    Task(
        object_color="blue",
        target_side="left",
        instruction="Pick up the blue cube and place it on the left.",
    ),
    Task(
        object_color="blue",
        target_side="right",
        instruction="Pick up the blue cube and place it on the right.",
    ),
]