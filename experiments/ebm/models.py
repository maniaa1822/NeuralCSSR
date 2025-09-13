import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        self.d_model = d_model
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Match positional encoding dtype/device to input (important for AMP/mixed precision)
        pe = self.pe[:x.size(0), :]
        if pe.dtype != x.dtype:
            pe = pe.to(dtype=x.dtype)
        if pe.device != x.device:
            pe = pe.to(device=x.device)
        return x + pe


class AutoRegressiveBinaryLM(nn.Module):
    def __init__(
        self,
        vocab_size: int = 3,
        output_vocab_size: int = 2,
        d_model: int = 128,
        nhead: int = 8,
        num_layers: int = 4,
        max_len: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.output_vocab_size = output_vocab_size

        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_len)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.dropout = nn.Dropout(dropout)

        self.lm_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, output_vocab_size),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        batch_size, seq_len = input_ids.shape
        x = self.token_embedding(input_ids) * math.sqrt(self.d_model)
        x = x.transpose(0, 1)
        x = self.positional_encoding(x)
        x = x.transpose(0, 1)
        x = self.dropout(x)
        # Causal mask (prevent attention to future positions)
        causal_mask = torch.ones(seq_len, seq_len, device=input_ids.device, dtype=torch.bool).triu(diagonal=1)
        padding_mask = (attention_mask == 0) if attention_mask is not None else None
        x = self.encoder(x, mask=causal_mask, src_key_padding_mask=padding_mask)
        logits = self.lm_head(x)
        return logits

    @torch.no_grad()
    def generate_probabilities(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        logits = self.forward(input_ids, attention_mask)
        if attention_mask is not None:
            last_idx = attention_mask.sum(dim=1) - 1
            batch_indices = torch.arange(logits.size(0), device=logits.device)
            last_logits = logits[batch_indices, last_idx]
        else:
            last_logits = logits[:, -1]
        probs = F.softmax(last_logits, dim=-1)
        return probs

    def compute_loss(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, target_ids: torch.Tensor) -> torch.Tensor:
        logits = self.forward(input_ids, attention_mask)
        if attention_mask is not None:
            last_idx = attention_mask.sum(dim=1) - 1
            batch_indices = torch.arange(logits.size(0), device=logits.device)
            pred_logits = logits[batch_indices, last_idx]
        else:
            pred_logits = logits[:, -1]
        loss = F.cross_entropy(pred_logits, target_ids)
        return loss


class EnergyBasedBinaryLM(nn.Module):
    def __init__(
        self,
        vocab_size: int = 3,
        output_vocab_size: int = 2,
        d_model: int = 128,
        nhead: int = 8,
        num_layers: int = 4,
        max_len: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.output_vocab_size = output_vocab_size

        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_len)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.dropout = nn.Dropout(dropout)

        self.energy_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, output_vocab_size),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        batch_size, seq_len = input_ids.shape
        x = self.token_embedding(input_ids) * math.sqrt(self.d_model)
        x = x.transpose(0, 1)
        x = self.positional_encoding(x)
        x = x.transpose(0, 1)
        x = self.dropout(x)
        # Causal mask (prevent attention to future positions)
        causal_mask = torch.ones(seq_len, seq_len, device=input_ids.device, dtype=torch.bool).triu(diagonal=1)
        padding_mask = (attention_mask == 0) if attention_mask is not None else None
        x = self.encoder(x, mask=causal_mask, src_key_padding_mask=padding_mask)
        energies = self.energy_head(x)
        scores = -energies
        return scores

    @torch.no_grad()
    def generate_probabilities(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        scores = self.forward(input_ids, attention_mask)
        if attention_mask is not None:
            last_idx = attention_mask.sum(dim=1) - 1
            batch_indices = torch.arange(scores.size(0), device=scores.device)
            last_scores = scores[batch_indices, last_idx]
        else:
            last_scores = scores[:, -1]
        probs = F.softmax(last_scores, dim=-1)
        return probs

    def compute_loss(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, target_ids: torch.Tensor) -> torch.Tensor:
        scores = self.forward(input_ids, attention_mask)
        if attention_mask is not None:
            last_idx = attention_mask.sum(dim=1) - 1
            batch_indices = torch.arange(scores.size(0), device=scores.device)
            pred_scores = scores[batch_indices, last_idx]
        else:
            pred_scores = scores[:, -1]
        loss = F.cross_entropy(pred_scores, target_ids)
        return loss


