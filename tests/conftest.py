"""Offline test doubles: a random-weight encoder and a hash tokenizer.

Nothing here touches the network - the whole training/evaluation pipeline is
exercised with locally constructed stand-ins for the HF model and tokenizer.
"""

from types import SimpleNamespace

import pytest
import torch
from torch import nn

VOCAB_SIZE = 128


class DummyTokenizer:
    """Deterministic hash tokenizer with the subset of the HF API we call."""

    model_max_length = 512

    def __init__(self, vocab_size: int = VOCAB_SIZE):
        self.vocab_size = vocab_size

    def __call__(
        self,
        texts,
        max_length=None,
        truncation=True,
        padding=True,
        return_tensors="pt",
        **kwargs,
    ):
        max_length = max_length or self.model_max_length
        # 2 reserved ids: 0 = pad, 1 = [CLS]
        sequences = [
            [1]
            + [2 + (hash(word) % (self.vocab_size - 2)) for word in text.split()][
                : max_length - 1
            ]
            for text in texts
        ]
        longest = max(len(s) for s in sequences)
        input_ids = torch.zeros(len(sequences), longest, dtype=torch.long)
        attention_mask = torch.zeros(len(sequences), longest, dtype=torch.long)
        for row, seq in enumerate(sequences):
            input_ids[row, : len(seq)] = torch.tensor(seq)
            attention_mask[row, : len(seq)] = 1
        return {"input_ids": input_ids, "attention_mask": attention_mask}

    def save_pretrained(self, path):
        import pathlib

        pathlib.Path(path).mkdir(parents=True, exist_ok=True)


class DummyEncoder(nn.Module):
    """A tiny randomly initialised transformer-shaped encoder."""

    def __init__(self, hidden_size: int = 32, num_hidden_layers: int = 4):
        super().__init__()
        self.config = SimpleNamespace(
            hidden_size=hidden_size,
            num_hidden_layers=num_hidden_layers,
            vocab_size=VOCAB_SIZE,
        )
        self.embeddings = nn.Embedding(VOCAB_SIZE, hidden_size)
        self.layers = nn.ModuleList(
            [nn.Linear(hidden_size, hidden_size) for _ in range(num_hidden_layers)]
        )
        self.mixers = nn.ModuleList(
            [nn.Linear(hidden_size, hidden_size) for _ in range(num_hidden_layers)]
        )
        self.dropout = nn.Dropout(0.1)  # gives SimCSE its two different views

    def forward(
        self,
        input_ids,
        attention_mask=None,
        output_hidden_states=False,
        return_dict=True,
        **kw,
    ):
        hidden = self.dropout(self.embeddings(input_ids))
        states = [hidden]
        if attention_mask is None:
            attention_mask = torch.ones(input_ids.shape, dtype=torch.long, device=input_ids.device)
        mask = attention_mask.unsqueeze(-1).to(hidden.dtype)

        for layer, mixer in zip(self.layers, self.mixers):
            # Cheap stand-in for attention: every token also sees the sequence mean,
            # so the [CLS] position actually carries sentence-level information.
            context = (hidden * mask).sum(dim=1, keepdim=True) / mask.sum(dim=1, keepdim=True).clamp(min=1)
            hidden = torch.tanh(layer(self.dropout(hidden)) + mixer(context))
            states.append(hidden)
        return SimpleNamespace(last_hidden_state=hidden, hidden_states=tuple(states))

    def save_pretrained(self, path):
        import pathlib

        pathlib.Path(path).mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), pathlib.Path(path) / "pytorch_model.bin")


@pytest.fixture
def offline_backbone(monkeypatch):
    """Patch the trainer's model/tokenizer loaders so no download happens."""
    from embedding_mrl.trainers import base

    monkeypatch.setattr(
        base.AutoTokenizer, "from_pretrained", lambda *a, **k: DummyTokenizer()
    )
    monkeypatch.setattr(
        base.AutoModel,
        "from_pretrained",
        lambda *a, **k: DummyEncoder(hidden_size=32, num_hidden_layers=4),
    )
    return DummyEncoder
