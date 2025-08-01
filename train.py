#!/usr/bin/env python3
"""
Training script for transformer models on binary sequence prediction.

This script handles all training logic with support for:
- Multiple model architectures
- Config file or CLI arguments  
- Comprehensive logging and checkpointing
- Early stopping and learning rate scheduling

Usage:
    python train.py --config configs/streamlined_golden_mean.yaml
    python train.py --model streamlined --train domain_machines/golden_mean/golden_mean.dat
"""

import argparse
import json
import math
import yaml
from pathlib import Path
from typing import Dict, Any, Optional

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from models.transformer_models import create_model, count_parameters
from models.data_utils import create_dataloaders


class TrainingConfig:
    """Configuration container for training parameters."""
    
    def __init__(self, **kwargs):
        # Model parameters
        self.model_type = kwargs.get('model_type', 'streamlined')
        self.vocab_size = kwargs.get('vocab_size', 3)
        self.d_model = kwargs.get('d_model', 64)
        self.n_layers = kwargs.get('n_layers', 2)
        self.n_heads = kwargs.get('n_heads', 4)
        self.dropout = kwargs.get('dropout', 0.1)
        self.delay = kwargs.get('delay', 10)  # For time-delay models
        self.window_size = kwargs.get('window_size', 25)  # For sliding window models
        
        # Training parameters
        self.epochs = kwargs.get('epochs', 10)
        self.batch_size = kwargs.get('batch_size', 32)
        self.lr = kwargs.get('lr', 1e-3)
        self.weight_decay = kwargs.get('weight_decay', 0.01)
        self.grad_clip = kwargs.get('grad_clip', 1.0)
        self.warmup_steps = kwargs.get('warmup_steps', 0)
        
        # Data parameters
        self.chunk_size = kwargs.get('chunk_size', 25)
        self.sliding_window = kwargs.get('sliding_window', True)  # Default to sliding window
        self.test_split = kwargs.get('test_split', 0.2)
        
        # Paths
        self.train_path = Path(kwargs.get('train_path', ''))
        self.test_path = Path(kwargs.get('test_path', '')) if kwargs.get('test_path') else None
        self.output_dir = Path(kwargs.get('output_dir', 'checkpoints/default'))
        
        # Other
        self.device = kwargs.get('device', 'auto')
        self.save_every = kwargs.get('save_every', 5)
        self.early_stopping_patience = kwargs.get('early_stopping_patience', 10)

    @classmethod
    def from_yaml(cls, config_path: Path):
        """Load configuration from YAML file."""
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        # Flatten nested structure
        flattened = {}
        
        # Model parameters
        if 'model' in config_dict:
            model_config = config_dict['model']
            flattened.update({
                'model_type': model_config.get('type', 'streamlined'),
                'vocab_size': model_config.get('vocab_size', 3),
                'd_model': model_config.get('d_model', 64),
                'n_layers': model_config.get('n_layers', 2),
                'n_heads': model_config.get('n_heads', 4),
                'dropout': model_config.get('dropout', 0.1),
                'delay': model_config.get('delay', 10),
                'window_size': model_config.get('window_size', 25),
            })
        
        # Training parameters
        if 'training' in config_dict:
            train_config = config_dict['training']
            flattened.update({
                'epochs': train_config.get('epochs', 10),
                'batch_size': train_config.get('batch_size', 32),
                'lr': train_config.get('learning_rate', 1e-3),
                'weight_decay': train_config.get('weight_decay', 0.01),
                'grad_clip': train_config.get('gradient_clipping', 1.0),
            })
        
        # Data parameters
        if 'data' in config_dict:
            data_config = config_dict['data']
            flattened.update({
                'train_path': data_config.get('train_path', ''),
                'test_path': data_config.get('test_path'),
                'chunk_size': data_config.get('chunk_size', 25),
                'sliding_window': data_config.get('sliding_window', True),
                'test_split': data_config.get('test_split', 0.2),
            })
        
        # Experiment parameters
        if 'experiment' in config_dict:
            exp_config = config_dict['experiment']
            flattened.update({
                'experiment_name': exp_config.get('name', 'experiment'),
                'output_dir': exp_config.get('output_dir', 'checkpoints/'),
            })
        
        return cls(**flattened)

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary for serialization."""
        return {k: str(v) if isinstance(v, Path) else v for k, v in self.__dict__.items()}


def train_epoch(model, loader, optimizer, scheduler, device, loss_fn, grad_clip=1.0):
    """Train model for one epoch."""
    model.train()
    total_loss = 0
    total_correct = 0
    total_tokens = 0
    
    for batch_idx, (input_ids, target_ids) in enumerate(loader):
        input_ids, target_ids = input_ids.to(device), target_ids.to(device)
        
        # Forward pass
        logits = model(input_ids)  # [B, T-1, vocab_size]
        
        # Compute loss
        loss = loss_fn(logits.reshape(-1, logits.size(-1)), target_ids.reshape(-1))
        
        # Backward pass with gradient clipping
        optimizer.zero_grad()
        loss.backward()
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        
        # Statistics
        total_loss += loss.item()
        predictions = logits.argmax(dim=-1)
        total_correct += (predictions == target_ids).sum().item()
        total_tokens += target_ids.numel()
        
        # Periodic logging
        if batch_idx % max(1, len(loader) // 10) == 0:
            acc = total_correct / total_tokens if total_tokens > 0 else 0
            lr = optimizer.param_groups[0]['lr']
            print(f"  Batch {batch_idx:3d}/{len(loader):3d} | "
                  f"Loss: {loss.item():.4f} | Acc: {acc:.3f} | LR: {lr:.6f}")
    
    avg_loss = total_loss / len(loader)
    accuracy = total_correct / total_tokens if total_tokens > 0 else 0
    perplexity = math.exp(avg_loss)
    
    return avg_loss, accuracy, perplexity


def evaluate_model(model, loader, device, loss_fn):
    """Evaluate model on test set."""
    model.eval()
    total_loss = 0
    total_correct = 0
    total_tokens = 0
    
    with torch.no_grad():
        for input_ids, target_ids in loader:
            input_ids, target_ids = input_ids.to(device), target_ids.to(device)
            
            logits = model(input_ids)
            loss = loss_fn(logits.reshape(-1, logits.size(-1)), target_ids.reshape(-1))
            
            total_loss += loss.item()
            predictions = logits.argmax(dim=-1)
            total_correct += (predictions == target_ids).sum().item()
            total_tokens += target_ids.numel()
    
    avg_loss = total_loss / len(loader)
    accuracy = total_correct / total_tokens if total_tokens > 0 else 0
    perplexity = math.exp(avg_loss)
    
    return avg_loss, accuracy, perplexity


def save_checkpoint(model, optimizer, epoch, config, metrics, output_dir):
    """Save training checkpoint."""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'config': config.to_dict(),
        'model_config': model.get_config(),
        'metrics': metrics,
    }
    
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / f'checkpoint_epoch_{epoch}.pt'
    torch.save(checkpoint, checkpoint_path)
    
    # Also save as 'latest.pt'
    latest_path = output_dir / 'latest.pt'
    torch.save(checkpoint, latest_path)
    
    return checkpoint_path


def main():
    parser = argparse.ArgumentParser(description='Train transformer models on binary sequences')
    
    # Config file or individual arguments
    parser.add_argument('--config', type=Path, help='YAML config file')
    
    # Model arguments
    parser.add_argument('--model', choices=['streamlined', 'time_delay'], default='streamlined')
    parser.add_argument('--d_model', type=int, default=64)
    parser.add_argument('--layers', type=int, default=2)
    parser.add_argument('--heads', type=int, default=4)
    parser.add_argument('--dropout', type=float, default=0.1)
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--grad_clip', type=float, default=1.0)
    
    # Data arguments
    parser.add_argument('--train', type=Path, help='Training .dat file')
    parser.add_argument('--test', type=Path, help='Test .dat file (optional)')
    parser.add_argument('--chunk_size', type=int, default=25)
    parser.add_argument('--test_split', type=float, default=0.2)
    
    # Output arguments
    parser.add_argument('--output_dir', type=Path, default=Path('checkpoints/default'))
    parser.add_argument('--save_every', type=int, default=5)
    
    args = parser.parse_args()
    
    # Load configuration
    if args.config:
        config = TrainingConfig.from_yaml(args.config)
        print(f"Loaded config from {args.config}")
    else:
        # Use CLI arguments
        config = TrainingConfig(
            model_type=args.model,
            d_model=args.d_model,
            n_layers=args.layers,
            n_heads=args.heads,
            dropout=args.dropout,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            weight_decay=args.weight_decay,
            grad_clip=args.grad_clip,
            train_path=args.train,
            test_path=args.test,
            chunk_size=args.chunk_size,
            test_split=args.test_split,
            output_dir=args.output_dir,
            save_every=args.save_every,
        )
    
    # Set device
    if config.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(config.device)
    
    print("=" * 70)
    print("🚀 NEURAL CSSR TRANSFORMER TRAINING")
    print("=" * 70)
    print(f"Model: {config.model_type}")
    print(f"Data: {config.train_path}")
    print(f"Output: {config.output_dir}")
    print(f"Device: {device}")
    print()
    
    # Create model
    model = create_model(
        config.model_type,
        vocab_size=config.vocab_size,
        d_model=config.d_model,
        n_layers=config.n_layers,
        n_heads=config.n_heads,
        dropout=config.dropout,
        delay=config.delay if config.model_type == 'time_delay' else None
    )
    model.to(device)
    
    # Count parameters
    total_params, trainable_params = count_parameters(model)
    print(f"Model parameters: {total_params:,} total, {trainable_params:,} trainable")
    print()
    
    # Create data loaders
    train_loader, test_loader = create_dataloaders(
        config.train_path,
        config.test_path,
        batch_size=config.batch_size,
        chunk_size=config.chunk_size,
        test_split=config.test_split,
        sliding_window=config.sliding_window,
    )
    print()
    
    # Training setup
    optimizer = AdamW(
        model.parameters(), 
        lr=config.lr, 
        weight_decay=config.weight_decay,
        betas=(0.9, 0.98)  # From original streamlined transformer
    )
    
    scheduler = CosineAnnealingLR(optimizer, T_max=config.epochs * len(train_loader))
    loss_fn = nn.CrossEntropyLoss(ignore_index=2)  # Ignore padding tokens
    
    # Training loop
    print(f"Starting training for {config.epochs} epochs...")
    print("-" * 70)
    
    best_test_ppl = float('inf')
    training_history = []
    
    for epoch in range(1, config.epochs + 1):
        print(f"Epoch {epoch:2d}/{config.epochs}")
        
        # Training
        train_loss, train_acc, train_ppl = train_epoch(
            model, train_loader, optimizer, scheduler, device, loss_fn, config.grad_clip
        )
        
        # Evaluation
        test_loss, test_acc, test_ppl = evaluate_model(model, test_loader, device, loss_fn)
        
        # Metrics
        metrics = {
            'epoch': epoch,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'train_ppl': train_ppl,
            'test_loss': test_loss,
            'test_acc': test_acc,
            'test_ppl': test_ppl,
            'lr': optimizer.param_groups[0]['lr']
        }
        training_history.append(metrics)
        
        # Print results
        print(f"  Train: loss={train_loss:.4f}, acc={train_acc:.3f}, ppl={train_ppl:.3f}")
        print(f"  Test:  loss={test_loss:.4f}, acc={test_acc:.3f}, ppl={test_ppl:.3f}")
        
        # Save best model
        if test_ppl < best_test_ppl:
            best_test_ppl = test_ppl
            print(f"  🎯 New best model! (test ppl: {test_ppl:.3f})")
            
            # Save best checkpoint
            best_path = save_checkpoint(model, optimizer, epoch, config, metrics, config.output_dir)
            best_link = config.output_dir / 'best.pt'
            if best_link.exists():
                best_link.unlink()
            best_link.symlink_to(best_path.name)
        
        # Periodic saves
        if epoch % config.save_every == 0:
            save_checkpoint(model, optimizer, epoch, config, metrics, config.output_dir)
        
        print()
    
    # Save training history
    history_path = config.output_dir / 'training_history.json'
    with open(history_path, 'w') as f:
        json.dump(training_history, f, indent=2)
    
    print("-" * 70)
    print(f"✅ Training complete! Best test perplexity: {best_test_ppl:.3f}")
    print(f"📁 Checkpoints saved to: {config.output_dir}")
    print(f"📊 Training history: {history_path}")


if __name__ == '__main__':
    main()