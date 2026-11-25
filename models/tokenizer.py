
import re

INSTRUCTIONS = [
    "Pick up the red cube and place it on the left.",
    "Pick up the red cube and place it on the right.",
    "Pick up the blue cube and place it on the left.",
    "Pick up the blue cube and place it on the right.",
]

words = sorted(set(
    word
    for sentence in INSTRUCTIONS
    for word in re.findall(r"[a-z]+", sentence.lower())
))

VOCAB = {"<pad>": 0, "<unk>": 1}
VOCAB.update({word: i + 2 for i, word in enumerate(words)})

MAX_LENGTH = 16


def tokenize(instruction):
    words = re.findall(r"[a-z]+", instruction.lower())

    ids = [VOCAB.get(word, 1) for word in words[:MAX_LENGTH]]

    return ids + [0] * (MAX_LENGTH - len(ids))
