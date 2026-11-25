from isaacsim import SimulationApp

app = SimulationApp({"headless": False})

from isaacsim.core.utils.extensions import enable_extension
enable_extension("isaacsim.robot.manipulators.examples")

from envs.franka_language_env import FrankaLanguageEnv
from controllers.expert import ExpertController


def main():
    env = FrankaLanguageEnv()

    for task_id in range(4):
        task = env.reset(task_id=task_id)

        expert = ExpertController(env.franka)
        expert.reset()

        success = expert.execute(env, task)

        print(f"Task {task_id}: {'SUCCESS' if success else 'FAILED'}")

        for _ in range(60):
            env.step(render=True)

    while app.is_running():
        env.step(render=True)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback
        traceback.print_exc()
    finally:
        app.close()