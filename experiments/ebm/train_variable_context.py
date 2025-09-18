#!/usr/bin/env python3
"""
Train neural models with variable context lengths to improve CSSR performance.
This addresses the issue where models trained with fixed context windows perform
poorly on the short histories typically used in CSSR analysis.
"""

import argparse
from pathlib import Path
import random
import math
import csv
from typing import List, Tuple, Dict
import numpy as np

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns

try:
    from .models import AutoRegressiveBinaryLM  # when executed as a module
except ImportError:
    import sys
    sys.path.append(str(Path(__file__).resolve().parent))
    from models import AutoRegressiveBinaryLM  # when executed as a script

def load_binary_tokens(dat_path: Path) -> List[int]:
    """Load binary sequence from .dat file."""
    content = dat_path.read_text().strip()
    tokens = [int(c) for c in content if c in '01']
    if not tokens:
        raise ValueError(f"No binary tokens found in {dat_path}")
    return tokens

def get_variable_batch(data_tokens: torch.Tensor, batch_size: int, max_context: int,
                      min_context: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Generate batch with variable context lengths.

    Returns:
        x: Input sequences [batch_size, max_context]
        y: Target tokens [batch_size]
        lengths: Actual sequence lengths [batch_size]
    """
    batch_x, batch_y, batch_lengths = [], [], []

    for _ in range(batch_size):
        # Random context length for this sequence
        seq_len = torch.randint(min_context, max_context + 1, (1,)).item()

        # Random starting position
        start_idx = torch.randint(0, len(data_tokens) - seq_len - 1, (1,)).item()

        # Extract sequence
        x_seq = data_tokens[start_idx:start_idx + seq_len]
        y_target = data_tokens[start_idx + seq_len]

        # Left-pad with zeros to max_context length
        if seq_len < max_context:
            pad_len = max_context - seq_len
            x_seq = torch.cat([torch.zeros(pad_len, dtype=x_seq.dtype), x_seq])

        batch_x.append(x_seq)
        batch_y.append(y_target)
        batch_lengths.append(seq_len)

    x = torch.stack(batch_x).to(device)
    y = torch.stack(batch_y).to(device)
    lengths = torch.tensor(batch_lengths, dtype=torch.long, device=device)

    return x, y, lengths

def create_length_aware_mask(lengths: torch.Tensor, max_len: int) -> torch.Tensor:
    """Create attention mask based on actual sequence lengths."""
    batch_size = lengths.size(0)
    mask = torch.zeros((batch_size, max_len), dtype=torch.bool, device=lengths.device)

    for i, length in enumerate(lengths):
        # Allow attention to actual sequence content (right-aligned)
        start_pos = max_len - length
        mask[i, start_pos:] = True

    return mask

@torch.no_grad()
def evaluate_by_context_length(model: AutoRegressiveBinaryLM, tokens: List[int],
                              device: torch.device, max_context: int,
                              steps_per_length: int = 50) -> Dict[int, float]:
    """Evaluate model performance by context length."""
    model.eval()
    data_tensor = torch.tensor(tokens, dtype=torch.long)

    results = {}

    for context_len in range(2, max_context + 1, 2):  # Test every 2 lengths
        losses = []

        for _ in range(steps_per_length):
            # Generate batch with fixed context length
            batch_x, batch_y = [], []

            for _ in range(32):  # Small batch for evaluation
                start_idx = torch.randint(0, len(data_tensor) - context_len - 1, (1,)).item()
                x_seq = data_tensor[start_idx:start_idx + context_len]
                y_target = data_tensor[start_idx + context_len]

                # Pad to max_context
                if context_len < max_context:
                    pad_len = max_context - context_len
                    x_seq = torch.cat([torch.zeros(pad_len, dtype=x_seq.dtype), x_seq])

                batch_x.append(x_seq)
                batch_y.append(y_target)

            x = torch.stack(batch_x).to(device)
            y = torch.stack(batch_y).to(device)

            # Forward pass
            logits = model(x)  # AutoRegressiveBinaryLM returns logits directly
            last_logits = logits[:, -1, :]  # Last position logits
            loss = F.cross_entropy(last_logits, y)
            losses.append(loss.item())

        results[context_len] = np.mean(losses)
        print(f"Context length {context_len}: Loss = {results[context_len]:.4f}")

    return results

def analyze_attention_patterns(model: AutoRegressiveBinaryLM, tokens: List[int],
                             device: torch.device, context_len: int,
                             max_context: int, save_path: Path) -> Dict:
    """
    Analyze attention patterns to see if model learns proper state dependencies.
    """
    model.eval()
    data_tensor = torch.tensor(tokens, dtype=torch.long)

    # Hook to capture attention weights
    attention_weights = []

    def attention_hook(module, input, output):
        # output is (attn_output, attn_weights)
        if len(output) > 1:
            attn = output[1]  # [batch, heads, seq_len, seq_len]
            attention_weights.append(attn.detach().cpu())

    # Register hooks on all attention layers
    hooks = []
    for name, module in model.named_modules():
        if hasattr(module, 'self_attn') or 'attention' in name.lower():
            hooks.append(module.register_forward_hook(attention_hook))

    try:
        # Generate sample with specific context length
        start_idx = torch.randint(0, len(data_tensor) - context_len - 1, (1,)).item()
        x_seq = data_tensor[start_idx:start_idx + context_len]

        # Pad to model's max context
        max_ctx = max_context
        if context_len < max_ctx:
            pad_len = max_ctx - context_len
            x_seq = torch.cat([torch.zeros(pad_len, dtype=x_seq.dtype), x_seq])

        x = x_seq.unsqueeze(0).to(device)

        # Forward pass to capture attention
        with torch.no_grad():
            logits = model(x)

        # Analyze captured attention weights
        if attention_weights:
            # Average across layers and heads
            avg_attention = torch.stack(attention_weights).mean(dim=[0, 1, 2])  # [seq_len, seq_len]

            # Focus on the last position (where prediction happens)
            last_pos_attention = avg_attention[-1, :]  # Attention from last position to all positions

            # Plot attention pattern
            plt.figure(figsize=(12, 8))

            plt.subplot(2, 2, 1)
            plt.imshow(avg_attention.numpy(), cmap='Blues')
            plt.title(f'Average Attention Pattern (Context Length {context_len})')
            plt.xlabel('Source Position')
            plt.ylabel('Target Position')
            plt.colorbar()

            plt.subplot(2, 2, 2)
            plt.plot(last_pos_attention.numpy())
            plt.title('Attention from Last Position')
            plt.xlabel('Source Position')
            plt.ylabel('Attention Weight')
            plt.axvline(x=max_ctx - context_len, color='red', linestyle='--',
                       label=f'Actual sequence start')
            plt.legend()

            # Analyze attention focus
            actual_start = max_ctx - context_len
            attention_on_sequence = last_pos_attention[actual_start:].sum().item()
            attention_on_padding = last_pos_attention[:actual_start].sum().item()

            plt.subplot(2, 2, 3)
            plt.bar(['Padding', 'Sequence'], [attention_on_padding, attention_on_sequence])
            plt.title('Attention Distribution')
            plt.ylabel('Total Attention Weight')

            # State transition analysis for 7-state machine
            sequence_part = x_seq[actual_start:].cpu().numpy()
            state_positions = []

            # Find positions where state might change (simplified heuristic)
            for i in range(1, len(sequence_part)):
                if i >= 2:  # Need at least 2 symbols to determine some states
                    state_positions.append(actual_start + i)

            plt.subplot(2, 2, 4)
            seq_attention = last_pos_attention[actual_start:].numpy()
            plt.plot(range(len(seq_attention)), seq_attention, 'b-', alpha=0.7)
            for pos in state_positions:
                rel_pos = pos - actual_start
                if rel_pos < len(seq_attention):
                    plt.axvline(x=rel_pos, color='red', alpha=0.3, linestyle=':')
            plt.title('Attention on Sequence (State Transitions Marked)')
            plt.xlabel('Position in Sequence')
            plt.ylabel('Attention Weight')

            plt.tight_layout()
            plt.savefig(save_path / f'attention_analysis_ctx{context_len}.png',
                       dpi=150, bbox_inches='tight')
            plt.close()

            return {
                'context_length': context_len,
                'attention_on_sequence': attention_on_sequence,
                'attention_on_padding': attention_on_padding,
                'sequence_focus_ratio': attention_on_sequence / (attention_on_sequence + attention_on_padding),
                'avg_attention_matrix': avg_attention.numpy(),
                'last_position_attention': last_pos_attention.numpy()
            }

    finally:
        # Remove hooks
        for hook in hooks:
            hook.remove()

    return {}

def train_with_variable_context(args):
    """Main training function with variable context lengths."""

    # Setup device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)

    print(f"Using device: {device}")

    # Load data
    tokens = load_binary_tokens(args.data)
    print(f"Loaded {len(tokens)} tokens from {args.data}")

    # Split train/val
    split_idx = int(len(tokens) * (1 - args.val_frac))
    train_tokens = torch.tensor(tokens[:split_idx], dtype=torch.long)
    val_tokens = torch.tensor(tokens[split_idx:], dtype=torch.long)

    print(f"Train tokens: {len(train_tokens)}, Val tokens: {len(val_tokens)}")

    # Create model
    model_config = {
        'vocab_size': 3,  # 0, 1, padding
        'output_vocab_size': 2,  # 0, 1
        'd_model': args.d_model,
        'nhead': args.heads,
        'num_layers': args.layers,
        'max_len': args.max_context,
        'dropout': args.dropout
    }

    model = AutoRegressiveBinaryLM(**model_config)
    model.to(device)

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scaler = torch.cuda.amp.GradScaler() if args.amp and device.type == 'cuda' else None

    # Training loop
    model.train()
    step = 0
    train_losses = []

    print(f"\nTraining with variable context lengths ({args.min_context}-{args.max_context})")
    print(f"Total steps: {args.epochs * args.steps_per_epoch}")

    for epoch in range(args.epochs):
        epoch_losses = []

        for step_in_epoch in range(args.steps_per_epoch):
            optimizer.zero_grad()

            # Get variable context batch
            x, y, lengths = get_variable_batch(
                train_tokens, args.batch_size, args.max_context,
                args.min_context, device
            )

            # Create attention mask for variable lengths
            attention_mask = create_length_aware_mask(lengths, args.max_context)

            # Forward pass
            if scaler is not None:
                with torch.amp.autocast('cuda'):
                    logits = model(x, attention_mask)  # Returns logits directly
                    # Get logits for last valid position for each sequence
                    last_idx = lengths - 1  # 0-indexed
                    batch_indices = torch.arange(logits.size(0), device=logits.device)
                    last_logits = logits[batch_indices, args.max_context - 1]  # Always last position since we right-pad
                    loss = F.cross_entropy(last_logits, y)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                logits = model(x, attention_mask)
                last_logits = logits[:, -1, :]  # Last position
                loss = F.cross_entropy(last_logits, y)

                loss.backward()
                optimizer.step()

            train_losses.append(loss.item())
            epoch_losses.append(loss.item())
            step += 1

            if step % 100 == 0:
                avg_loss = np.mean(train_losses[-100:])
                avg_length = lengths.float().mean().item()
                print(f"Step {step}: Loss = {avg_loss:.4f}, Avg Context = {avg_length:.1f}")

        # Evaluate on validation set
        model.eval()
        val_losses = []

        with torch.no_grad():
            for _ in range(args.val_steps):
                x, y, lengths = get_variable_batch(
                    val_tokens, args.batch_size, args.max_context,
                    args.min_context, device
                )

                attention_mask = create_length_aware_mask(lengths, args.max_context)
                logits = model(x, attention_mask)
                last_logits = logits[:, -1, :]
                loss = F.cross_entropy(last_logits, y)
                val_losses.append(loss.item())

        train_loss = np.mean(epoch_losses)
        val_loss = np.mean(val_losses)

        print(f"Epoch {epoch + 1}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}")

        model.train()

    # Final evaluation by context length
    print("\nEvaluating performance by context length...")
    length_performance = evaluate_by_context_length(
        model, tokens, device, args.max_context
    )

    # Analyze attention patterns
    print("Analyzing attention patterns...")
    attention_dir = args.ckpt_out.parent / 'attention_analysis'
    attention_dir.mkdir(exist_ok=True)

    attention_results = {}
    for ctx_len in [2, 4, 8, 16, 32]:
        if ctx_len <= args.max_context:
            result = analyze_attention_patterns(
                model, tokens, device, ctx_len, args.max_context, attention_dir
            )
            if result:
                attention_results[ctx_len] = result

    # Save model
    model_state = {
        'state_dict': model.state_dict(),
        'config': model_config,
        'train_losses': train_losses,
        'length_performance': length_performance,
        'attention_analysis': attention_results,
        'args': vars(args)
    }

    torch.save(model_state, args.ckpt_out)
    print(f"Model saved to {args.ckpt_out}")

    # Save results to CSV
    results_csv = args.ckpt_out.parent / f'{args.ckpt_out.stem}_results.csv'
    with open(results_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['context_length', 'loss', 'sequence_focus_ratio'])

        for ctx_len, loss in length_performance.items():
            focus_ratio = attention_results.get(ctx_len, {}).get('sequence_focus_ratio', 0.0)
            writer.writerow([ctx_len, loss, focus_ratio])

    print(f"Results saved to {results_csv}")

    return model, length_performance, attention_results

def main():
    parser = argparse.ArgumentParser(description="Train with variable context lengths")

    # Data and model
    parser.add_argument('--data', type=Path, required=True, help='Path to .dat file')
    parser.add_argument('--preset', type=str, default='seven_state_human',
                       choices=['golden_mean', 'even_process', 'seven_state_human'])

    # Architecture
    parser.add_argument('--max_context', type=int, default=32,
                       help='Maximum context window length')
    parser.add_argument('--min_context', type=int, default=2,
                       help='Minimum context window length')
    parser.add_argument('--d_model', type=int, default=128)
    parser.add_argument('--layers', type=int, default=4)
    parser.add_argument('--heads', type=int, default=8)
    parser.add_argument('--dropout', type=float, default=0.0)

    # Training
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--steps_per_epoch', type=int, default=500)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--val_frac', type=float, default=0.1)
    parser.add_argument('--val_steps', type=int, default=100)

    # System
    parser.add_argument('--device', type=str, default='auto')
    parser.add_argument('--amp', action='store_true', help='Use mixed precision')
    parser.add_argument('--seed', type=int, default=42)

    # Output
    parser.add_argument('--ckpt_out', type=Path,
                       default=Path('/home/matteo/NeuralCSSR/experiments/ebm/checkpoints/variable_context_model.pt'))

    args = parser.parse_args()

    # Set default data path if not specified
    if not args.data:
        if args.preset == 'seven_state_human':
            args.data = Path('/home/matteo/NeuralCSSR/experiments/datasets/seven_state_human/seven_state_human.dat')
        elif args.preset == 'golden_mean':
            args.data = Path('/home/matteo/NeuralCSSR/experiments/datasets/golden_mean/golden_mean.dat')
        elif args.preset == 'even_process':
            args.data = Path('/home/matteo/NeuralCSSR/experiments/datasets/even_process/even_process.dat')

    # Set seeds
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Create output directory
    args.ckpt_out.parent.mkdir(parents=True, exist_ok=True)

    # Train model
    model, length_performance, attention_results = train_with_variable_context(args)

    print("\n🎯 TRAINING COMPLETE!")
    print(f"✅ Model saved to: {args.ckpt_out}")
    print(f"📊 Performance by context length:")
    for ctx_len, loss in sorted(length_performance.items()):
        print(f"   Context {ctx_len:2d}: Loss = {loss:.4f}")

    print(f"🔍 Attention analysis saved to: {args.ckpt_out.parent / 'attention_analysis'}")

if __name__ == "__main__":
    main()