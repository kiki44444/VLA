
import numpy as np
import torch
from models.vla_bc import MLPVLA

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = MLPVLA().to(device)
model.load_state_dict(torch.load("checkpoints/vla_bc.pt", map_location=device, weights_only=True))
model.eval()

data = np.load("data/demos/task_0_ep_000.npz")

images = data["images"]
states = data["robot_states"]
actions = data["actions"]

# Inspect steps around grasp, transport, and release
steps = [50, 55, 59, 60, 61, 65, 70,
         90, 95, 99, 100, 101, 105, 110,
         150, 155, 159, 160, 161, 165, 170]

print("Step | XYZ Error (mm) | True Grip | Pred Grip")

with torch.no_grad():
    for step in steps:
        image = torch.from_numpy(images[step].copy()).permute(2, 0, 1)
        image = image.unsqueeze(0).float().to(device) / 255.0

        state = torch.tensor(states[step:step+1], dtype=torch.float32, device=device)
        task_id = torch.tensor([0], dtype=torch.long, device=device)

        pred = model(image, state, task_id)[0]

        xyz = pred[:3].cpu().numpy()
        grip_prob = torch.sigmoid(pred[3]).item()

        xyz_error = np.linalg.norm(xyz - actions[step, :3]) * 1000

        print(
            f"{step:03d}  | {xyz_error:8.1f}       | "
            f"{int(actions[step, 3])}         | {grip_prob:.3f}"
        )
