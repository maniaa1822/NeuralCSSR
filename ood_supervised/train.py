from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from .losses import compute_losses


@dataclass
class TrainingConfig:
    epochs: int = 5
    batch_size: int = 16
    learning_rate: float = 1e-4
    row_entropy_coeff: float = 0.1
    label_smoothing: float = 0.0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    log_interval: Optional[int] = 10
    checkpoint_path: Optional[Path] = None
    log_dir: Optional[Path] = None


def train_model(model, dataset, config: TrainingConfig, collate_fn) -> Dict[str, float]:
    if isinstance(dataset, tuple):
        train_dataset, val_dataset = dataset
    else:
        train_dataset, val_dataset = dataset, None

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = None
    if val_dataset is not None:
        val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, collate_fn=collate_fn)
    model.to(config.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)

    logger = logging.getLogger(__name__)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
        logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    logger.info(
        "Starting training: epochs=%s, train_batches=%s, val_batches=%s, device=%s",
        config.epochs,
        len(train_loader),
        len(val_loader) if val_loader else 0,
        config.device,
    )

    writer: Optional[SummaryWriter] = None
    if config.log_dir:
        config.log_dir.mkdir(parents=True, exist_ok=True)
        writer = SummaryWriter(log_dir=str(config.log_dir))

    num_batches = len(train_loader)

    for epoch in range(config.epochs):
        model.train()
        epoch_totals: Dict[str, float] = defaultdict(float)
        running_totals: Dict[str, float] = defaultdict(float)
        log_interval = config.log_interval if config.log_interval and config.log_interval > 0 else None
        for batch_idx, batch in enumerate(train_loader, start=1):
            tokens = batch["tokens"].to(config.device)
            attention_mask = batch["attention_mask"].to(config.device)
            emissions = batch["emissions"].to(config.device)
            transitions = batch["transitions"].to(config.device)
            mask = batch["mask"].to(config.device)
            pred_emissions, pred_transitions, pred_mask_logits = model(tokens, attention_mask=attention_mask)

            outputs = {
                "emissions": pred_emissions,
                "transition_logits": pred_transitions,
                "mask_logits": pred_mask_logits,
            }
            targets = {
                "emissions": emissions,
                "transitions": transitions,
                "mask": mask,
            }

            losses = compute_losses(
                outputs,
                targets,
                row_entropy_coeff=config.row_entropy_coeff,
                label_smoothing=config.label_smoothing,
            )

            optimizer.zero_grad()
            losses["total"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            for name, value in losses.items():
                scalar = float(value.item())
                epoch_totals[name] += scalar
                if log_interval:
                    running_totals[name] += scalar
                if writer is not None:
                    global_step = epoch * num_batches + (batch_idx - 1)
                    writer.add_scalar(f"loss/{name}", scalar, global_step)

            if log_interval and batch_idx % log_interval == 0:
                msg_parts = [
                    f"{name}={running_totals[name] / log_interval:.4f}"
                    for name in sorted(running_totals.keys())
                ]
                logger.info(
                    "Epoch %d/%d Step %d/%d | %s",
                    epoch + 1,
                    config.epochs,
                    batch_idx,
                    num_batches,
                    ", ".join(msg_parts),
                )
                running_totals = defaultdict(float)

        epoch_means = {
            name: value / max(num_batches, 1)
            for name, value in epoch_totals.items()
        }
        logger.info(
            "Epoch %d summary | %s",
            epoch + 1,
            ", ".join(f"{name}={value:.4f}" for name, value in sorted(epoch_means.items())),
        )
        if writer is not None:
            for name, value in epoch_means.items():
                writer.add_scalar(f"epoch/{name}", value, epoch + 1)

        # validation pass
        if val_loader is not None:
            model.eval()
            with torch.no_grad():
                val_totals: Dict[str, float] = defaultdict(float)
                for val_batch in val_loader:
                    tokens = val_batch["tokens"].to(config.device)
                    attention_mask = val_batch["attention_mask"].to(config.device)
                    emissions = val_batch["emissions"].to(config.device)
                    transitions = val_batch["transitions"].to(config.device)
                    mask = val_batch["mask"].to(config.device)

                    pred_emissions, pred_transitions, pred_mask_logits = model(tokens, attention_mask=attention_mask)

                    outputs = {
                        "emissions": pred_emissions,
                        "transition_logits": pred_transitions,
                        "mask_logits": pred_mask_logits,
                    }
                    targets = {
                        "emissions": emissions,
                        "transitions": transitions,
                        "mask": mask,
                    }

                    losses = compute_losses(
                        outputs,
                        targets,
                        row_entropy_coeff=config.row_entropy_coeff,
                        label_smoothing=0.0,
                    )

                    for name, value in losses.items():
                        val_totals[name] += float(value.item())

                val_means = {
                    name: value / max(len(val_loader), 1)
                    for name, value in val_totals.items()
                }
                logger.info(
                    "Epoch %d validation | %s",
                    epoch + 1,
                    ", ".join(f"{name}={value:.4f}" for name, value in sorted(val_means.items())),
                )
                if writer is not None:
                    for name, value in val_means.items():
                        writer.add_scalar(f"val/{name}", value, epoch + 1)
        last_epoch_means = epoch_means
    if config.checkpoint_path:
        config.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), config.checkpoint_path)
        logger.info("Saved checkpoint to %s", config.checkpoint_path)

    if writer is not None:
        writer.flush()
        writer.close()

    return last_epoch_means
