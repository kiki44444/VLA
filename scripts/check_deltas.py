
import numpy as np

data = np.load("data/demos/task_0_ep_000.npz")
actions = data["actions"].astype(np.float32)

targets = actions[:, :3]
deltas = np.diff(targets, axis=0, prepend=targets[:1])

magnitudes = np.linalg.norm(deltas, axis=1)

print("Frames:", len(deltas))
print("Zero deltas:", np.mean(magnitudes < 1e-6) * 100, "%")
print("Max delta:", magnitudes.max(), "m")
print("Mean delta:", magnitudes.mean(), "m")

for step in [0, 40, 60, 70, 80, 100, 120, 140, 160, 180]:
    print(f"Step {step:03d} | Delta: {np.round(deltas[step], 4)}")
