
import numpy as np

from isaacsim.core.utils.rotations import euler_angles_to_quat
from isaacsim.robot.manipulators.examples.franka.controllers import RMPFlowController


class ExpertController:
    def __init__(self, franka):
        self.franka = franka
        self.controller = RMPFlowController(
            name="expert_rmpflow",
            robot_articulation=franka
        )
        self.orientation = euler_angles_to_quat(
            np.array([0.0, np.pi, 0.0])
        )

        self.control_target = None
        self.max_delta = 0.02


    def reset(self):
        self.controller.reset()
        self.franka.gripper.open()
        self.control_target = None

    def move_to(self, target):
        action = self.controller.forward(
            target_end_effector_position=np.asarray(target),
            target_end_effector_orientation=self.orientation,
        )
        self.franka.apply_action(action)

    def move_incremental(self, target):
        target = np.asarray(target, dtype=np.float64)

        delta = np.clip(
            target - self.control_target,
            -self.max_delta,
            self.max_delta
        )

        self.control_target = self.control_target + delta
        self.move_to(self.control_target)

        return delta.copy()

    def get_plan(self, task):
        obj = np.array(task["target_position"])
        goal = np.array(task["goal_position"])

        return [
            (obj + [0, 0, 0.15], 1, 180),
            (obj + [0, 0, 0.03], 1, 180),
            (obj + [0, 0, 0.03], 0, 60),
            (obj + [0, 0, 0.20], 0, 180),
            (goal + [0, 0, 0.20], 0, 180),
            (goal + [0, 0, 0.03], 0, 180),
            (goal + [0, 0, 0.03], 1, 60),
            (goal + [0, 0, 0.20], 1, 180),
        ]
