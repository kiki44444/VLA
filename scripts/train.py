
from pathlib import Path
from datetime import datetime
import random
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

from models.vla_bc import MLPVLA


class DemoDataset(Dataset):
    def __init__(self, paths):
        self.episodes = []

        for path in paths:
            with np.load(path) as data:
                self.episodes.append({
                    "images": data["images"],
                    "states": data["robot_states"],
                    "control_targets": data["control_targets"],
                    "gripper_states": data["gripper_states"],
                    "actions": data["actions"],
                    "task_id": int(path.stem.split("_")[1]),
                })

        self.indices = [
            (i, t)
            for i, ep in enumerate(self.episodes)
            for t in range(len(ep["actions"]))
        ]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        i, t = self.indices[idx]
        ep = self.episodes[i]

        image = torch.from_numpy(ep["images"][t].copy()).permute(2, 0, 1).float() / 255.0
        state = torch.tensor(
            np.concatenate([
                ep["states"][t],
                ep["control_targets"][t],
                [ep["gripper_states"][t]]
            ]),
            dtype=torch.float32
        )
        action = torch.tensor(ep["actions"][t], dtype=torch.float32)
        task_id = torch.tensor(ep["task_id"], dtype=torch.long)

        return image, state, task_id, action


def compute_loss(pred, action):
    pos_loss = F.mse_loss(pred[:, :3],action[:, :3] * 100.0)
    grip_loss = F.binary_cross_entropy_with_logits(pred[:, 3],action[:, 3])
    return pos_loss + grip_loss, pos_loss, grip_loss


def run_epoch(model, loader, device, optimizer=None):
    training = optimizer is not None
    model.train(training)
    totals = np.zeros(3)
    samples = 0

    with torch.set_grad_enabled(training):
        for images, states, task_ids, actions in loader:
            images = images.to(device)
            states = states.to(device)
            task_ids = task_ids.to(device)
            actions = actions.to(device)

            pred = model(images, states, task_ids)
            loss, pos_loss, grip_loss = compute_loss(pred, actions)

            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            totals += np.array([loss.item(), pos_loss.item(), grip_loss.item()]) * len(images)
            samples += len(images)

    return totals / samples


def main():
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    paths = sorted(Path("data/demos_delta").glob("*.npz"))
    train_paths, val_paths = [], []

    for task_id in range(4):
        task_paths = [p for p in paths if int(p.stem.split("_")[1]) == task_id]
        if len(task_paths) < 2:
            raise ValueError(f"Task {task_id} needs at least 2 episodes")

        random.shuffle(task_paths)
        n_val = max(1, round(len(task_paths) * 0.2))
        val_paths += task_paths[:n_val]
        train_paths += task_paths[n_val:]

    train_loader = DataLoader(DemoDataset(train_paths), batch_size=32, shuffle=True)
    val_loader = DataLoader(DemoDataset(val_paths), batch_size=32, shuffle=False)

    print(f"Train: {len(train_paths)} episodes | Val: {len(val_paths)} episodes")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MLPVLA().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    Path("checkpoints").mkdir(exist_ok=True)

    run_name = datetime.now().strftime("%Y%m%d_%H%M%S")
    writer = SummaryWriter(f"runs/mlp_vla/{run_name}")

    best_val_loss = float("inf")

    try:
        for epoch in range(1, 21):
            train_loss, train_pos, train_grip = run_epoch(model, train_loader, device, optimizer)
            val_loss, val_pos, val_grip = run_epoch(model, val_loader, device)

            writer.add_scalars("Loss/Total", {"train": train_loss, "val": val_loss}, epoch)
            writer.add_scalars("Loss/Position", {"train": train_pos, "val": val_pos}, epoch)
            writer.add_scalars("Loss/Gripper", {"train": train_grip, "val": val_grip}, epoch)

            print(f"Epoch {epoch:02d} | Train: {train_loss:.4f} | Val: {val_loss:.4f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), "checkpoints/vla_bc_delta.pt")
    finally:
        writer.close()

    print(f"Best validation loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    main()
