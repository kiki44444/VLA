from isaacsim import SimulationApp

app = SimulationApp({"headless": False})

from isaacsim.core.utils.extensions import enable_extension

enable_extension("isaacsim.robot.manipulators.examples")

from envs.franka_language_env import FrankaLanguageEnv
from envs.camera import create_camera


def main():
    env = FrankaLanguageEnv()
    env.reset(task_id=0)

    camera = create_camera()

    for _ in range(60):
        env.step(render=True)

    rgb = camera.get_rgb()
    
    from PIL import Image
    Image.fromarray(rgb).save("camera_test.png")
    print("Saved camera_test.png")
    print("RGB shape:", rgb.shape)


if __name__ == "__main__":
    try:
        main()
    finally:
        app.close()