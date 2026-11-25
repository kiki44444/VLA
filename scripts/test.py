import torch

from models.vla_transformer import VLATransformer
from models.tokenizer import VOCAB, tokenize

model = VLATransformer(vocab_size=len(VOCAB))

image = torch.randn(2, 3, 256, 256)
state = torch.randn(2, 13)

instruction = torch.tensor([
    tokenize("Pick up the red cube and place it on the left."),
    tokenize("Pick up the blue cube and place it on the right.")
])

with torch.no_grad():
    action = model(image, state, instruction)

print(action.shape)