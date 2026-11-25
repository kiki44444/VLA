'''
Creates
1. Robot
2. Objects (Cube)
3. Goal positions
4. Task selection
'''
import numpy as np

from isaacsim.core.api import World
from isaacsim.core.api.objects import DynamicCuboid, VisualCuboid
from isaacsim.robot.manipulators.examples.franka import Franka

from configs.task_config import TASKS


class FrankaLanguageEnv:

    def __init__(self):
        self.world = World(stage_units_in_meters=1.0, physics_dt=1.0 / 60.0)
        self.world.scene.add_default_ground_plane()

        # Franka Panda
        self.franka = self.world.scene.add(Franka(prim_path="/World/Franka", name="franka"))

        # Cube configs
        self.cube_positions = {
            "red": np.array([0.50, -0.12, 0.025]),
            "blue": np.array([0.50, 0.12, 0.025]),
        }

        self.goal_positions = {
            "left": np.array([0.40, 0.30, 0.025]),
            "right": np.array([0.40, -0.30, 0.025]),
        }

        self.cubes = {}

        colors = {
            "red": np.array([1.0, 0.0, 0.0]),
            "blue": np.array([0.0, 0.0, 1.0]),
        }

        # Create two dynamic cubes
        for color, position in self.cube_positions.items():
            cube = self.world.scene.add(
                DynamicCuboid(
                    prim_path=f"/World/{color}_cube",
                    name=f"{color}_cube",
                    position=position,
                    scale=np.array([0.05, 0.05, 0.05]),
                    color=colors[color],
                    mass=0.05,
                )
            )

            self.cubes[color] = cube

        # Visual markers for the two goal positions
        for side, position in self.goal_positions.items():
            self.world.scene.add(
                VisualCuboid(
                    prim_path=f"/World/{side}_goal",
                    name=f"{side}_goal",
                    position=np.array(
                        [position[0], position[1], 0.002]
                    ),
                    scale=np.array([0.10, 0.10, 0.004]),
                    color=np.array([0.0, 1.0, 0.0]),
                )
            )

        self.current_task = None

    def reset(self, task_id=0):
        if not 0 <= task_id < len(TASKS):
            raise ValueError(f"Invalid task_id: {task_id}")

        self.world.reset()

        # Restore cube positions after reset
        for color, cube in self.cubes.items():
            cube.set_world_pose(position=self.cube_positions[color].copy())
            cube.set_linear_velocity(np.zeros(3))
            cube.set_angular_velocity(np.zeros(3))

        self.current_task = TASKS[task_id]

        print(f"Instruction: {self.current_task.instruction}")

        return self.get_task_info()

    def get_task_info(self):
        if self.current_task is None:
            raise RuntimeError("Call reset() first.")

        task = self.current_task

        return {
            "instruction": task.instruction,
            "target_object": task.object_color,
            "target_position": self.cube_positions[
                task.object_color
            ].copy(),
            "goal_position": self.goal_positions[
                task.target_side
            ].copy(),
        }

    def step(self, render=True):
        self.world.step(render=render)

    def randomize_cubes(self):
        for color, cube in self.cubes.items():
            x = np.random.uniform(0.45, 0.55)

            if color == "red":
                y = np.random.uniform(-0.18, -0.06)
            else:
                y = np.random.uniform(0.06, 0.18)

            position = np.array([x, y, 0.025])

            cube.set_world_pose(position=position)
            self.cube_positions[color] = position.copy()
