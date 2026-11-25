
import re
from pathlib import Path

import matplotlib.pyplot as plt

log_path = Path("logs/train.log")
output_path = Path("plots/training_loss.png")

epochs = []
losses = []

with open(log_path, "r") as f:
    for line in f:
        match = re.search(r"Epoch\s+(\d+)\s*\|\s*Loss:\s*([\d.]+)", line)

        if match:
            epochs.append(int(match.group(1)))
            losses.append(float(match.group(2)))

if not epochs:
    raise ValueError(f"No training losses found in {log_path}")

plt.figure(figsize=(8, 5))
plt.plot(epochs, losses, marker="o", markersize=4)

plt.xlabel("Epoch")
plt.ylabel("Training Loss")
plt.title("MLPVLA Training Loss")
plt.xticks(range(1, max(epochs) + 1, 2))
plt.grid(True, alpha=0.3)
plt.tight_layout()

output_path.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(output_path, dpi=300)
plt.show()
