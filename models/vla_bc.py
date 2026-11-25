import torch
import torch.nn as nn

class MLPVLA(nn.Module):
    def __init__(self):
        super().__init__()

        self.vision = nn.Sequential(
            nn.Conv2d(3, 32, 5, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )

        self.state_encoder = nn.Sequential(
            nn.Linear(13, 64),
            nn.ReLU()
        )
        
        self.language = nn.Embedding(4, 32)

        self.policy = nn.Sequential(
            nn.Linear(128 + 64 + 32, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 4),
        )

    def forward(self, image, state, task_id):
        image_features = self.vision(image)
        state_features = self.state_encoder(state)
        language_features = self.language(task_id)

        features = torch.cat(
            [image_features, state_features, language_features],
            dim=-1,
        )

        return self.policy(features)