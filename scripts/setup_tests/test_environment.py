from isaacsim import SimulationApp
app = SimulationApp({"headless": False})

from isaacsim.core.utils.extensions import enable_extension
enable_extension("isaacsim.robot.manipulators.examples")

from envs.franka_language_env import FrankaLanguageEnv

def main():
    env = FrankaLanguageEnv()
    task = env.reset(task_id=0)

    print(task)

    while app.is_running():
        env.step(render=True)

if __name__ == "__main__":
    try:
        main()
    finally:
        app.close()