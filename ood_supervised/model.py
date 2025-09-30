from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn


@dataclass
class DecoderConfig:
    vocab_size: int = 2
    max_length: int = 512
    hidden_size: int = 256
    num_layers: int = 4
    num_heads: int = 8
    mlp_dim: int = 512
    dropout: float = 0.1
    max_states: int = 12


class PositionalEncoding(nn.Module):
    def __init__(self, hidden_size: int, max_len: int = 2048):
        super().__init__()
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, hidden_size, 2) * (-torch.log(torch.tensor(10000.0)) / hidden_size))
        pe = torch.zeros(max_len, hidden_size)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


class EpsMachineDecoder(nn.Module):
    def __init__(self, config: DecoderConfig):
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        self.positional = PositionalEncoding(config.hidden_size, config.max_length)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.hidden_size,
            nhead=config.num_heads,
            dim_feedforward=config.mlp_dim,
            dropout=config.dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=config.num_layers)
        self.layer_norm = nn.LayerNorm(config.hidden_size)
        self.pooler = nn.Linear(config.hidden_size, config.hidden_size)
        self.pool_act = nn.Tanh()

        max_states = config.max_states
        hidden = config.hidden_size

        self.emission_head = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, max_states),
        )

        self.transition_head = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, max_states * 2 * max_states),
        )

        self.mask_head = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, max_states),
        )

    def forward(
        self,
        tokens: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        x = self.embedding(tokens)
        x = self.positional(x)
        x = self.layer_norm(x)
        x = self.encoder(x, src_key_padding_mask=~attention_mask if attention_mask is not None else None)

        if attention_mask is not None:
            masked = x * attention_mask.unsqueeze(-1)
            pooled = masked.sum(dim=1) / attention_mask.sum(dim=1, keepdim=True).clamp_min(1.0)
        else:
            pooled = x.mean(dim=1)

        pooled = self.pool_act(self.pooler(pooled))

        emissions = torch.sigmoid(self.emission_head(pooled))
        transitions = self.transition_head(pooled)
        transitions = transitions.view(-1, self.config.max_states, 2, self.config.max_states)
        mask_logits = self.mask_head(pooled)

        return emissions, transitions, mask_logits
