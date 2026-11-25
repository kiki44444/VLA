# Robot Manipulation with language

A VLA-inspired robot learning project using Isaac Sim 6.1, Franka Panda, and RMPFlow.

The goal is to train a language conditioned imitation learning model that picks up a specified object and places it at a target location.

## Task Setup

The environment contains two cubes (red and blue) and two target locations (left and right), resulting in four possible instructions.

## Expert

Implemented an RMPFlow expert that performs:
1. Approach the target cube
2. Grasp and lift
3. Move to the target location
4. Release and retreat

## Collecting demonstrations
1. Add camera envs/camera.py


## Troubleshooting
### 1. End ffector offset

The end effector position consistently differed from the target by around 5–7 cm.

The robot was moving to the correct position, but i thought the ee position indicated the center of the fingers but it was the right finger.

```text
Franka EE:  /World/Franka/panda_rightfinger
RMPFlow:    right_gripper
```

