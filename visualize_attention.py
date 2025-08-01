#!/usr/bin/env python3
"""
Simple Attention Pattern Visualization

This script visualizes attention patterns from a trained transformer model
on sequences from the dataset. Shows how the model attends to different
positions when processing sequential data.

Usage:
    python visualize_attention.py --checkpoint checkpoints/sliding_window_seven_state/best.pt
"""

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse
from typing import List, Tuple

from models.data_utils import SequenceDataset
from extract_fsm_sliding_window import load_model_from_checkpoint


def extract_attention_weights(model, input_ids: torch.Tensor) -> List[torch.Tensor]:
    """Extract attention weights from the model."""
    model.eval()
    attention_weights = []
    
    with torch.no_grad():
        # Handle different model architectures
        if hasattr(model, 'transformer'):
            # SlidingWindowTransformer uses nn.TransformerDecoder
            embeddings = model.embedding(input_ids)
            x = model.pos_encoder(embeddings)
            
            # For TransformerDecoder, we need to process differently
            # We'll extract attention by temporarily modifying the layers
            for layer in model.transformer.layers:
                # Extract attention weights from self-attention
                # Create causal mask
                seq_len = x.size(1)
                causal_mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool().to(x.device)
                causal_mask = causal_mask.masked_fill(causal_mask, float('-inf'))
                
                # Self-attention (decoder self-attention)
                attn_output, attn_weights = layer.self_attn(
                    query=x, key=x, value=x, 
                    attn_mask=causal_mask,
                    need_weights=True, average_attn_weights=False
                )
                attention_weights.append(attn_weights.cpu())
                
                # Continue forward pass
                x = layer.norm1(x + layer.dropout1(attn_output))
                ff_output = layer.linear2(layer.dropout(layer.activation(layer.linear1(x))))
                x = layer.norm2(x + layer.dropout2(ff_output))
        
        elif hasattr(model, 'transformer_decoder'):
            # StreamlinedTransformer
            embeddings = model.embedding(input_ids)
            x = embeddings
            
            # Get attention from each layer
            for layer in model.transformer_decoder.layers:
                # Extract attention weights
                attn_output, attn_weights = layer.self_attn(
                    x, x, x, need_weights=True, average_attn_weights=False
                )
                attention_weights.append(attn_weights.cpu())
                
                # Continue forward pass
                x = layer.norm1(x + attn_output)
                x = layer.norm2(x + layer.linear2(F.relu(layer.linear1(x))))
        
        elif hasattr(model, 'layers'):
            # TimeDelayTransformer
            x = model.embedding(input_ids)
            for layer in model.layers:
                if hasattr(layer, 'self_attn'):
                    attn_output, attn_weights = layer.self_attn(
                        x, x, x, need_weights=True, average_attn_weights=False
                    )
                    attention_weights.append(attn_weights.cpu())
                    x = layer.norm1(x + attn_output)
                    x = layer.norm2(x + layer.feed_forward(x))
        
        else:
            print("⚠️ Model architecture not recognized for attention extraction")
            print(f"   Available attributes: {[attr for attr in dir(model) if not attr.startswith('_')]}")
            return []
    
    return attention_weights


def visualize_attention_patterns(tokens: List[int], attention_weights: List[torch.Tensor], 
                                sequence_idx: int = 0, save_path: Path = None):
    """Visualize attention patterns for a sequence."""
    
    if not attention_weights:
        print("❌ No attention weights available")
        return
    
    n_layers = len(attention_weights)
    n_heads = attention_weights[0].shape[1] if len(attention_weights[0].shape) >= 2 else 1
    
    print(f"📊 Visualizing attention: {n_layers} layers, {n_heads} heads per layer")
    
    # Create subplots
    fig, axes = plt.subplots(n_layers, 1, figsize=(12, 4 * n_layers))
    if n_layers == 1:
        axes = [axes]
    
    fig.suptitle(f'Attention Patterns - Sequence {sequence_idx}\nTokens: {tokens}', 
                fontsize=14, fontweight='bold')
    
    for layer_idx, layer_attn in enumerate(attention_weights):
        # Average over heads and batch dimension
        if len(layer_attn.shape) == 4:  # [batch, heads, seq, seq]
            avg_attn = layer_attn[0].mean(dim=0).numpy()  # Average over heads
        elif len(layer_attn.shape) == 3:  # [heads, seq, seq]
            avg_attn = layer_attn.mean(dim=0).numpy()
        else:
            avg_attn = layer_attn.numpy()
        
        # Create heatmap
        im = axes[layer_idx].imshow(avg_attn, cmap='Blues', aspect='auto', vmin=0, vmax=1)
        axes[layer_idx].set_title(f'Layer {layer_idx + 1} (avg over {n_heads} heads)')
        axes[layer_idx].set_xlabel('Key Position')
        axes[layer_idx].set_ylabel('Query Position')
        
        # Add token labels
        token_labels = [str(t) for t in tokens]
        seq_len = len(token_labels)
        
        axes[layer_idx].set_xticks(range(seq_len))
        axes[layer_idx].set_yticks(range(seq_len))
        axes[layer_idx].set_xticklabels(token_labels, fontsize=10)
        axes[layer_idx].set_yticklabels(token_labels, fontsize=10)
        
        # Add colorbar
        plt.colorbar(im, ax=axes[layer_idx], fraction=0.046, pad=0.04)
        
        # Add grid for better readability
        axes[layer_idx].set_xticks(np.arange(-0.5, seq_len, 1), minor=True)
        axes[layer_idx].set_yticks(np.arange(-0.5, seq_len, 1), minor=True)
        axes[layer_idx].grid(which="minor", color="gray", linestyle='-', linewidth=0.5, alpha=0.3)
        
        # Highlight strong attention patterns
        max_attn = np.max(avg_attn)
        threshold = max_attn * 0.8  # Highlight top 20% of attention weights
        
        for i in range(seq_len):
            for j in range(seq_len):
                if avg_attn[i, j] > threshold:
                    axes[layer_idx].text(j, i, f'{avg_attn[i, j]:.2f}', 
                                        ha='center', va='center', 
                                        color='red', fontweight='bold', fontsize=8)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"💾 Attention patterns saved to {save_path}")
    
    plt.show()


def analyze_attention_head_specialization(attention_weights: List[torch.Tensor], tokens: List[int]):
    """Analyze what different attention heads focus on."""
    
    if not attention_weights:
        return
    
    print("🔍 Analyzing attention head specialization...")
    
    for layer_idx, layer_attn in enumerate(attention_weights):
        if len(layer_attn.shape) < 3:
            continue
            
        # Get attention for first sequence in batch
        if len(layer_attn.shape) == 4:
            attn = layer_attn[0]  # [heads, seq, seq]
        else:
            attn = layer_attn
        
        n_heads, seq_len, _ = attn.shape
        
        print(f"\nLayer {layer_idx + 1}:")
        
        for head_idx in range(n_heads):
            head_attn = attn[head_idx].numpy()
            
            # Analyze attention patterns for this head
            # 1. Self-attention strength (diagonal)
            self_attn = np.mean(np.diag(head_attn))
            
            # 2. Local vs global attention
            local_attn = 0.0  # Attention to adjacent positions
            for i in range(seq_len):
                for j in range(max(0, i-1), min(seq_len, i+2)):
                    if i != j:  # Exclude self-attention
                        local_attn += head_attn[i, j]
            local_attn /= (seq_len * 2)  # Normalize
            
            global_attn = np.sum(head_attn) - np.sum(np.diag(head_attn)) - local_attn * seq_len * 2
            global_attn /= (seq_len * (seq_len - 3))  # Normalize
            
            # 3. Token-specific patterns
            token_0_focus = 0.0
            token_1_focus = 0.0
            
            for i, token in enumerate(tokens):
                if token == 0:
                    token_0_focus += np.sum(head_attn[:, i]) / seq_len
                elif token == 1:
                    token_1_focus += np.sum(head_attn[:, i]) / seq_len
            
            print(f"  Head {head_idx + 1}:")
            print(f"    Self-attention: {self_attn:.3f}")
            print(f"    Local attention: {local_attn:.3f}")
            print(f"    Global attention: {global_attn:.3f}")
            print(f"    Focus on token 0: {token_0_focus:.3f}")
            print(f"    Focus on token 1: {token_1_focus:.3f}")


def create_attention_head_comparison(attention_weights: List[torch.Tensor], tokens: List[int], 
                                   sequence_idx: int = 0, save_path: Path = None):
    """Create side-by-side comparison of attention heads."""
    
    if not attention_weights:
        return
    
    # Focus on the first layer for detailed head analysis
    layer_attn = attention_weights[0]
    
    if len(layer_attn.shape) == 4:
        attn = layer_attn[0]  # [heads, seq, seq]
    else:
        attn = layer_attn
    
    n_heads = attn.shape[0]
    
    # Create subplots for each head
    fig, axes = plt.subplots(1, n_heads, figsize=(4 * n_heads, 4))
    if n_heads == 1:
        axes = [axes]
    
    fig.suptitle(f'Attention Head Comparison - Layer 1, Sequence {sequence_idx}\nTokens: {tokens}', 
                fontsize=14, fontweight='bold')
    
    for head_idx in range(n_heads):
        head_attn = attn[head_idx].numpy()
        
        im = axes[head_idx].imshow(head_attn, cmap='Blues', aspect='auto', vmin=0, vmax=1)
        axes[head_idx].set_title(f'Head {head_idx + 1}')
        axes[head_idx].set_xlabel('Key Position')
        axes[head_idx].set_ylabel('Query Position')
        
        # Add token labels
        token_labels = [str(t) for t in tokens]
        seq_len = len(token_labels)
        
        axes[head_idx].set_xticks(range(seq_len))
        axes[head_idx].set_yticks(range(seq_len))  
        axes[head_idx].set_xticklabels(token_labels, fontsize=10)
        axes[head_idx].set_yticklabels(token_labels, fontsize=10)
        
        # Add colorbar
        plt.colorbar(im, ax=axes[head_idx], fraction=0.046, pad=0.04)
        
        # Add grid
        axes[head_idx].set_xticks(np.arange(-0.5, seq_len, 1), minor=True)
        axes[head_idx].set_yticks(np.arange(-0.5, seq_len, 1), minor=True)
        axes[head_idx].grid(which="minor", color="gray", linestyle='-', linewidth=0.5, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"💾 Head comparison saved to {save_path}")
    
    plt.show()


def main():
    parser = argparse.ArgumentParser(description='Visualize attention patterns from trained transformer')
    parser.add_argument('--checkpoint', type=Path, required=True, help='Model checkpoint path')
    parser.add_argument('--data', type=Path, 
                       default=Path('domain_machines/seven_state_human/seven_state_human/seven_state_human.dat'),
                       help='Data file path')
    parser.add_argument('--output_dir', type=Path, default=Path('attention_visualizations'),
                       help='Output directory for plots')
    parser.add_argument('--num_sequences', type=int, default=5,
                       help='Number of sequences to analyze')
    parser.add_argument('--sequence_length', type=int, default=25,
                       help='Length of sequences to analyze')
    
    args = parser.parse_args()
    
    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load model
    print("Loading model...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = load_model_from_checkpoint(args.checkpoint, device)
    
    # Load data
    print("Loading data...")
    dataset = SequenceDataset(args.data, chunk_size=args.sequence_length, sliding_window=True)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)
    
    print(f"\n{'='*60}")
    print("🎯 ATTENTION PATTERN ANALYSIS")
    print(f"{'='*60}")
    
    # Analyze multiple sequences
    for seq_idx, batch in enumerate(dataloader):
        if seq_idx >= args.num_sequences:
            break
        
        input_ids = batch.to(device)
        tokens = input_ids[0].cpu().numpy().tolist()
        
        print(f"\n🔍 Analyzing sequence {seq_idx + 1}:")
        print(f"   Tokens: {tokens}")
        print(f"   Length: {len(tokens)}")
        
        # Extract attention weights
        attention_weights = extract_attention_weights(model, input_ids)
        
        if not attention_weights:
            print(f"   ❌ Failed to extract attention for sequence {seq_idx + 1}")
            continue
        
        # Visualize attention patterns
        save_path = args.output_dir / f'attention_sequence_{seq_idx + 1}.png'
        visualize_attention_patterns(tokens, attention_weights, seq_idx + 1, save_path)
        
        # Create head comparison for first sequence
        if seq_idx == 0:
            head_save_path = args.output_dir / f'attention_heads_sequence_{seq_idx + 1}.png'
            create_attention_head_comparison(attention_weights, tokens, seq_idx + 1, head_save_path)
        
        # Analyze head specialization
        analyze_attention_head_specialization(attention_weights, tokens)
    
    print(f"\n{'='*60}")
    print("✅ ATTENTION ANALYSIS COMPLETE")
    print(f"{'='*60}")
    print(f"Visualizations saved to: {args.output_dir}")


if __name__ == '__main__':
    main()