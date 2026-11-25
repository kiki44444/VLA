
from isaacsim import SimulationApp

app = SimulationApp({"headless": False})

from isaacsim.core.utils.extensions import enable_extension
enable_extension("isaacsim.robot.manipulators.examples")

import numpy as np
import torch

from envs.franka_language_env import FrankaLanguageEnv
from envs.camera import create_camera
from controllers.expert import ExpertController
from models.vla_bc import MLPVLA


def predict_action(model, camera, env, task_id, control_target, current_gripper, device):
    image = torch.from_numpy(camera.get_rgb().copy()).permute(2, 0, 1)
    image = image.unsqueeze(0).float().to(device) / 255.0

    state = np.concatenate([
        env.franka.get_joint_positions(),
        control_target,
        [current_gripper]
    ])

    state = torch.tensor(state[None], dtype=torch.float32, device=device)
    task = torch.tensor([task_id], dtype=torch.long, device=device)

    with torch.no_grad():
        pred = model(image, state, task)[0]

    delta = pred[:3].cpu().numpy() / 100.0
    delta = np.clip(delta, -0.02, 0.02)

    gripper = torch.sigmoid(pred[3]).item()

    return delta, gripper


def evaluate_task(env, camera, controller, model, task_id, device):
    task = env.reset(task_id=task_id)
    controller.reset()

    for _ in range(30):
        env.step(render=True)

    # Match the initial controller target used during collection
    control_target = np.asarray(
        controller.get_plan(task)[0][0], dtype=np.float64
    ).copy()

    current_gripper = 1.0

    cube = env.cubes[task["target_object"]]
    goal = np.array(task["goal_position"])

    max_height = 0.0

    for step in range(200):
        delta, gripper = predict_action(
            model, camera, env, task_id,
            control_target, current_gripper, device
        )

        control_target = np.clip(
            control_target + delta,
            [0.25, -0.35, 0.035],
            [0.75, 0.45, 0.50]
        )

        grip_open = float(gripper >= 0.5)

        if grip_open != current_gripper:
            if grip_open:
                env.franka.gripper.open()
            else:
                env.franka.gripper.close()

            current_gripper = grip_open

        for _ in range(6):
            controller.move_to(control_target)
            env.step(render=True)

        cube_pos, _ = cube.get_world_pose()
        max_height = max(max_height, cube_pos[2])

        if step % 20 == 0:
            print(
                f"Step {step:03d} | "
                f"Target: {np.round(control_target, 3)} | "
                f"Delta: {np.round(delta, 3)} | "
                f"Gripper: {gripper:.2f}"
            )

    cube_pos, _ = cube.get_world_pose()

    xy_error = np.linalg.norm(cube_pos[:2] - goal[:2])
    lifted = max_height > 0.10
    placed = xy_error < 0.05 and abs(cube_pos[2] - goal[2]) < 0.03
    success = lifted and placed

    print(
        f"Task {task_id} | "
        f"Lifted: {lifted} | "
        f"XY error: {xy_error:.4f} | "
        f"Success: {success}"
    )

    return success


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = MLPVLA().to(device)
    model.load_state_dict(
        torch.load(
            "checkpoints/vla_bc_delta.pt",
            map_location=device,
            weights_only=True
        )
    )
    model.eval()

    env = FrankaLanguageEnv()
    env.reset(task_id=0)

    camera = create_camera()
    controller = ExpertController(env.franka)

    results = []

    for task_id in range(4):
        success = evaluate_task(
            env, camera, controller, model, task_id, device
        )
        results.append(success)

    print(f"\nSuccess rate: {sum(results)}/{len(results)}")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback
        traceback.print_exc()
    finally:
        app.close()
