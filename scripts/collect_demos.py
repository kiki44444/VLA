
from isaacsim import SimulationApp

app = SimulationApp({"headless": False})

from isaacsim.core.utils.extensions import enable_extension
enable_extension("isaacsim.robot.manipulators.examples")

import numpy as np
from pathlib import Path

from envs.franka_language_env import FrankaLanguageEnv
from envs.camera import create_camera
from controllers.expert import ExpertController


def collect_episode(env, expert, camera, task_id):
    task = env.reset(task_id=task_id)
    env.randomize_cubes()
    task = env.get_task_info()
    expert.reset()

    for _ in range(30):
        env.step(render=True)

    observations = []
    actions = []

    plan = expert.get_plan(task)

    # Initialize at the first waypoint.
    # This first positioning command is not included in the dataset.
    expert.control_target = np.asarray(plan[0][0], dtype=np.float64).copy()

    current_gripper = 1.0  # Initially open

    for target, gripper, steps in plan:
        for _ in range(steps // 6):
            rgb = camera.get_rgb().copy()
            joints = env.franka.get_joint_positions().copy()
            control_target = expert.control_target.copy()

            observations.append({
                "image": rgb,
                "robot_state": joints,
                "control_target": control_target,
                "gripper_state": current_gripper,
                "instruction": task["instruction"],
            })

            delta = np.clip(
                np.asarray(target) - control_target,
                -expert.max_delta,
                expert.max_delta
            )

            actions.append(np.concatenate([delta, [gripper]]))

            if gripper != current_gripper:
                if gripper:
                    expert.franka.gripper.open()
                else:
                    expert.franka.gripper.close()
                current_gripper = float(gripper)

            for _ in range(6):
                expert.move_to(control_target + delta)
                env.step(render=True)

            expert.control_target = control_target + delta

    cube_pos, _ = env.cubes[task["target_object"]].get_world_pose()
    goal = np.array(task["goal_position"])

    success = (
        np.linalg.norm(cube_pos[:2] - goal[:2]) < 0.05
        and abs(cube_pos[2] - goal[2]) < 0.03
    )

    return observations, np.array(actions), success


def main():
    env = FrankaLanguageEnv()
    camera = create_camera()
    expert = ExpertController(env.franka)

    output_dir = Path("data/demos_delta")
    output_dir.mkdir(parents=True, exist_ok=True)

    for task_id in range(4):
        for episode_id in range(20):
            observations, actions, success = collect_episode(
                env, expert, camera, task_id
            )

            if not success:
                print(f"Task {task_id}, Episode {episode_id}: FAILED")
                continue

            np.savez_compressed(
                output_dir / f"task_{task_id}_ep_{episode_id:03d}.npz",
                images=np.stack([obs["image"] for obs in observations]),
                robot_states=np.stack([obs["robot_state"] for obs in observations]),
                control_targets=np.stack([obs["control_target"] for obs in observations]),
                gripper_states=np.array(
                    [obs["gripper_state"] for obs in observations],
                    dtype=np.float32
                ),
                instructions=np.array(
                    [obs["instruction"] for obs in observations]
                ),
                actions=actions.astype(np.float32),
            )

            print(f"Task {task_id}, Episode {episode_id}: SUCCESS")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback
        traceback.print_exc()
    finally:
        app.close()
