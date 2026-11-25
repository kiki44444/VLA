
import torch
import torch.nn as nn


class VLATransformer(nn.Module):
    def __init__(self, vocab_size=32, state_dim=13, hidden_dim=256):
        super().__init__()

        # 1. Vision encoder
        self.vision_encoder = nn.Sequential(
            nn.Conv2d(3, 32, 5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, hidden_dim, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten()
        )

        # 2. Language encoder
        self.word_embedding = nn.Embedding(
            vocab_size, hidden_dim, padding_idx=0
        )

        # 3. Robot state encoder
        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, hidden_dim)
        )

        # 4. Learnable action token and modality embeddings
        self.action_token = nn.Parameter(
            torch.randn(1, 1, hidden_dim) * 0.02
        )

        self.modality_embedding = nn.Embedding(4, hidden_dim)

        # 5. Transformer
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=4,
            dim_feedforward=hidden_dim * 4,
            dropout=0.1,
            batch_first=True
        )

        self.transformer = nn.TransformerEncoder(
            layer,
            num_layers=4,
            enable_nested_tensor=False
        )

        # 6. Action prediction head
        self.action_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 4)
        )

    def forward(self, image, state, instruction):
        """
        image:       (B, 3, 256, 256)
        state:       (B, 13)
        instruction: (B, L) integer token IDs

        output:      (B, 4)
        """
        batch_size = image.shape[0]

        # Encode each modality
        vision = self.vision_encoder(image).unsqueeze(1)
        robot = self.state_encoder(state).unsqueeze(1)
        language = self.word_embedding(instruction)

        # Action token
        action = self.action_token.expand(batch_size, -1, -1)

        # Add modality embeddings
        vision = vision + self.modality_embedding.weight[0]
        language = language + self.modality_embedding.weight[1]
        robot = robot + self.modality_embedding.weight[2]
        action = action + self.modality_embedding.weight[3]

        # Combine tokens
        tokens = torch.cat(
            [action, vision, robot, language], dim=1
        )

        # Ignore padded language tokens
        padding_mask = torch.cat([
            torch.zeros(
                batch_size, 3, dtype=torch.bool,
                device=instruction.device
            ),
            instruction == 0
        ], dim=1)

        # Multimodal attention
        features = self.transformer(
            tokens,
            src_key_padding_mask=padding_mask
        )

        # Predict action from action token
        return self.action_head(features[:, 0])
