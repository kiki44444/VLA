# scripts/test_tasks.py
from configs.task_config import TASKS

def main():
    for i, task in enumerate(TASKS):
        print(f"Task {i}")
        print(f"  Instruction: {task.instruction}")
        print(f"  Object: {task.object_color}")
        print(f"  Target: {task.target_side}")

if __name__ == "__main__":
    main()