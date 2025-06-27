#!/usr/bin/env python3
"""
Training script for FSM Transformer following the fsm_transformer_plan.md

This script trains an autoregressive transformer on epsilon-machine generated sequences
to test whether the model can discover causal state structure through next-token prediction.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json
import argparse
from tqdm import tqdm
from typing import Dict, List, Optional

# Import FSM transformer modules
from fsm_transformer.transformer import AutoregressiveTransformer
from fsm_transformer.data_generator import load_dataset, EpsilonMachineDataGenerator
from fsm_transformer.epsilon_machine import EpsilonMachine
from fsm_transformer.analysis import CausalStateAnalyzer, run_full_analysis


class FSMTransformerTrainer:
    """Trainer for FSM Transformer experiment."""
    
    def __init__(
        self,
        model: AutoregressiveTransformer,
        dataset,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        warmup_steps: int = 1000,
        use_wandb: bool = False
    ):
        self.model = model.to(device)
        self.dataset = dataset
        self.device = device
        self.use_wandb = use_wandb
        
        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), 
            lr=lr, 
            weight_decay=weight_decay
        )
        
        # Learning rate scheduler with warmup
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer,
            lr_lambda=lambda step: min(1.0, step / warmup_steps) if step < warmup_steps else 0.5 ** ((step - warmup_steps) // 5000)
        )
        
        # Loss function
        self.criterion = nn.CrossEntropyLoss()
        
        # Training state
        self.step = 0
        self.epoch = 0
        self.best_loss = float('inf')
        
    def train_epoch(self, dataloader: DataLoader) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        correct_predictions = 0
        total_predictions = 0
        
        pbar = tqdm(dataloader, desc=f"Epoch {self.epoch}")
        
        for batch in pbar:
            # Move to device
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            target_ids = batch['target_id'].to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(input_ids, attention_mask)
            logits = outputs['logits']
            
            # Compute loss (predict next token from last position)
            last_logits = logits[:, -1, :]  # [batch_size, vocab_size]
            loss = self.criterion(last_logits, target_ids)
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            self.scheduler.step()
            
            # Track metrics
            total_loss += loss.item()
            
            # Accuracy
            predictions = torch.argmax(last_logits, dim=-1)
            correct_predictions += (predictions == target_ids).sum().item()
            total_predictions += target_ids.size(0)
            
            # Update progress bar
            accuracy = correct_predictions / total_predictions
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{accuracy:.4f}',
                'lr': f'{self.scheduler.get_last_lr()[0]:.6f}'
            })
            
            # Log metrics (wandb removed)
            pass
            
            self.step += 1
        
        epoch_loss = total_loss / len(dataloader)
        epoch_accuracy = correct_predictions / total_predictions
        
        return {
            'loss': epoch_loss,
            'accuracy': epoch_accuracy
        }
    
    def evaluate(self, dataloader: DataLoader) -> Dict[str, float]:
        """Evaluate model on validation set."""
        self.model.eval()
        total_loss = 0
        correct_predictions = 0
        total_predictions = 0
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Evaluating"):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                target_ids = batch['target_id'].to(self.device)
                
                outputs = self.model(input_ids, attention_mask)
                logits = outputs['logits']
                
                last_logits = logits[:, -1, :]
                loss = self.criterion(last_logits, target_ids)
                
                total_loss += loss.item()
                
                predictions = torch.argmax(last_logits, dim=-1)
                correct_predictions += (predictions == target_ids).sum().item()
                total_predictions += target_ids.size(0)
        
        return {
            'loss': total_loss / len(dataloader),
            'accuracy': correct_predictions / total_predictions
        }
    
    def save_checkpoint(self, path: str, metadata: Dict = None):
        """Save model checkpoint."""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'step': self.step,
            'epoch': self.epoch,
            'best_loss': self.best_loss
        }
        
        if metadata:
            checkpoint['metadata'] = metadata
        
        torch.save(checkpoint, path)
        print(f"Checkpoint saved to {path}")
    
    def load_checkpoint(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.step = checkpoint['step']
        self.epoch = checkpoint['epoch']
        self.best_loss = checkpoint['best_loss']
        
        print(f"Checkpoint loaded from {path}")


def create_dataloaders(dataset, batch_size: int = 32, train_split: float = 0.8):
    """Create train/validation dataloaders."""
    
    # Split dataset
    total_size = len(dataset)
    train_size = int(total_size * train_split)
    val_size = total_size - train_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=2,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=2,
        pin_memory=True
    )
    
    return train_loader, val_loader


def main():
    parser = argparse.ArgumentParser(description='Train FSM Transformer')
    parser.add_argument('--data_dir', type=str, default='data/fsm_transformer',
                       help='Directory containing dataset')
    parser.add_argument('--output_dir', type=str, default='fsm_transformer_results',
                       help='Directory to save results')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-3,
                       help='Learning rate')
    parser.add_argument('--embed_dim', type=int, default=64,
                       help='Embedding dimension')
    parser.add_argument('--num_heads', type=int, default=4,
                       help='Number of attention heads')
    parser.add_argument('--num_layers', type=int, default=4,
                       help='Number of transformer layers')
    parser.add_argument('--ff_dim', type=int, default=128,
                       help='Feed-forward dimension')
    parser.add_argument('--dropout', type=float, default=0.1,
                       help='Dropout rate')
    parser.add_argument('--warmup_steps', type=int, default=1000,
                       help='Warmup steps for learning rate')
    parser.add_argument('--save_every', type=int, default=10,
                       help='Save checkpoint every N epochs')
    parser.add_argument('--wandb', action='store_true',
                       help='Use Weights & Biases logging')
    parser.add_argument('--generate_data', action='store_true',
                       help='Generate new dataset instead of loading existing')
    parser.add_argument('--num_sequences', type=int, default=50000,
                       help='Number of sequences to generate (if --generate_data)')
    parser.add_argument('--run_analysis', action='store_true',
                       help='Run causal state analysis after training')
    
    args = parser.parse_args()
    
    # Create output directory
    output_path = Path(args.output_dir)
    output_path.mkdir(exist_ok=True, parents=True)
    
    # Initialize logging (wandb removed)
    pass
    
    # Load or generate dataset
    if args.generate_data:
        print("Generating new dataset...")
        generator = EpsilonMachineDataGenerator(seed=42)
        dataset, metadata = generator.generate_training_data(
            num_sequences=args.num_sequences,
            min_length=100,
            max_length=200,
            output_dir=args.data_dir
        )
    else:
        print(f"Loading dataset from {args.data_dir}...")
        dataset, metadata = load_dataset(args.data_dir)
    
    print(f"Dataset loaded: {len(dataset)} training examples")
    print(f"Vocabulary size: {dataset.vocab_size}")
    
    # Create model
    model = AutoregressiveTransformer(
        vocab_size=dataset.vocab_size,
        embed_dim=args.embed_dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        ff_dim=args.ff_dim,
        dropout=args.dropout,
        pad_token_id=dataset.pad_token_id
    )
    
    print(f"Model created: {sum(p.numel() for p in model.parameters())} parameters")
    
    # Create dataloaders
    train_loader, val_loader = create_dataloaders(dataset, args.batch_size)
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
    
    # Create trainer
    trainer = FSMTransformerTrainer(
        model=model,
        dataset=dataset,
        lr=args.lr,
        warmup_steps=args.warmup_steps,
        use_wandb=args.wandb
    )
    
    # Training loop
    train_losses = []
    val_losses = []
    val_accuracies = []
    
    print("Starting training...")
    for epoch in range(args.epochs):
        trainer.epoch = epoch
        
        # Train
        train_metrics = trainer.train_epoch(train_loader)
        train_losses.append(train_metrics['loss'])
        
        # Validate
        val_metrics = trainer.evaluate(val_loader)
        val_losses.append(val_metrics['loss'])
        val_accuracies.append(val_metrics['accuracy'])
        
        print(f"Epoch {epoch}: Train Loss: {train_metrics['loss']:.4f}, "
              f"Val Loss: {val_metrics['loss']:.4f}, Val Acc: {val_metrics['accuracy']:.4f}")
        
        # Log metrics (wandb removed)
        pass
        
        # Save checkpoint
        if val_metrics['loss'] < trainer.best_loss:
            trainer.best_loss = val_metrics['loss']
            trainer.save_checkpoint(
                str(output_path / 'best_model.pt'),
                metadata={'epoch': epoch, 'val_loss': val_metrics['loss']}
            )
        
        if (epoch + 1) % args.save_every == 0:
            trainer.save_checkpoint(str(output_path / f'checkpoint_epoch_{epoch}.pt'))
    
    # Save final model
    trainer.save_checkpoint(str(output_path / 'final_model.pt'))
    
    # Save training history
    history = {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'val_accuracies': val_accuracies,
        'args': vars(args)
    }
    
    with open(output_path / 'training_history.json', 'w') as f:
        json.dump(history, f, indent=2)
    
    # Plot training curves
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training Loss')
    
    plt.subplot(1, 3, 2)
    plt.plot(val_accuracies, label='Val Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Validation Accuracy')
    
    plt.subplot(1, 3, 3)
    epochs = list(range(len(train_losses)))
    plt.plot(epochs, train_losses, label='Train Loss')
    plt.plot(epochs, val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.yscale('log')
    plt.legend()
    plt.title('Training Loss (Log Scale)')
    
    plt.tight_layout()
    plt.savefig(output_path / 'training_curves.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Training complete! Results saved to {output_path}")
    print(f"Best validation loss: {trainer.best_loss:.4f}")
    print(f"Final validation accuracy: {val_accuracies[-1]:.4f}")
    
    # Run causal state analysis
    if args.run_analysis:
        print("Running causal state analysis...")
        analysis_results = run_full_analysis(
            model_path=str(output_path / 'best_model.pt'),
            data_path=args.data_dir,
            output_dir=str(output_path / 'analysis'),
            max_contexts=5000
        )
        
        print("Analysis complete!")
        print(f"State recovery ARI: {analysis_results['state_recovery']['adjusted_rand_score']:.4f}")
        print(f"Clustering purity: {analysis_results['state_recovery']['purity']:.4f}")
        print(f"Predictive accuracy: {analysis_results['predictive_accuracy']['overall_accuracy']:.4f}")
    
    # Wandb removed
    pass


if __name__ == "__main__":
    main()