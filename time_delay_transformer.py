import argparse
import math
import sys
from pathlib import Path
from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader

#usage uv run time_delay_transformer.py
#  --train datasets/biased_exp/neural_format/train_dataset.pt
#  --dev datasets/biased_exp/neural_format/val_dataset.pt
#  --mode ar --epochs 3 --batch 128 --d_model 10 --layers 1 --heads 1 --lr 1e-3

# Add src to path to import neural_cssr
sys.path.insert(0, str(Path(__file__).parent / "src"))

# ============================================================
# Dataset utilities
# ============================================================

class SequenceDataset(Dataset):
    """Loads binary symbol sequences (0/1) from a .pt file containing a list of tensors.

    Each item is a 1‑D LongTensor of shape [T].
    """

    def __init__(self, pt_file: Path):
        # Load with weights_only=False to handle custom classes
        sequences = torch.load(pt_file, weights_only=False)
        
        # Handle the NeuralCSSRDataset format
        if hasattr(sequences, 'examples'):
            # Extract input sequences from the dataset
            seq_list = []
            for example in sequences.examples:
                # Convert input_ids back to binary sequence
                input_ids = example['input_ids']
                # Remove padding tokens (token_id 0 = '<PAD>')
                # Correct token mapping: 0=PAD, 1=UNK, 2='0', 3='1'
                sequence = [token for token in input_ids if token != 0]  # Remove PAD tokens
                if len(sequence) > 1:  # Only add sequences with meaningful length
                    # Map token IDs: 2->'0' (maps to 0), 3->'1' (maps to 1)
                    mapped_sequence = []
                    for token in sequence:
                        if token == 2:  # '0' symbol
                            mapped_sequence.append(0)
                        elif token == 3:  # '1' symbol
                            mapped_sequence.append(1)
                        elif token == 1:  # UNK token, skip
                            print(f"Warning: UNK token found, skipping")
                            continue
                        else:
                            print(f"Warning: unexpected token {token}, skipping sequence")
                            break
                    else:  # Only executed if no break occurred
                        if len(mapped_sequence) > 1:  # Ensure we have a meaningful sequence
                            seq_list.append(torch.tensor(mapped_sequence, dtype=torch.long))
            self.data = seq_list
        elif isinstance(sequences, (list, tuple)):
            # Handle simple list format
            self.data = sequences
        else:
            raise ValueError("Expected a list[Tensor] or NeuralCSSRDataset in the .pt file")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def collate_fn_time_delay(batch, k: int):
    """Collate for time‑delay transformer.

    Given a batch of variable‑length sequences, pads them to the max length and
    returns (tokens, tgt) where:
        tokens: [B, T, k+1] with padded 2 for missing past positions.
        tgt   : [B, T]   next symbol labels.
    """
    max_len = max(seq.size(0) for seq in batch)
    B = len(batch)
    toks = torch.full((B, max_len, k + 1), 2, dtype=torch.long)  # Use 2 as padding
    tgt = torch.full((B, max_len), 2, dtype=torch.long)  # Use 2 as padding for targets too

    for b, seq in enumerate(batch):
        T = seq.size(0)
        # Build time‑delay embeddings
        for t in range(T):
            # history positions: t‑k .. t
            for d in range(k + 1):
                idx = t - (k - d)
                toks[b, t, d] = seq[idx] if idx >= 0 else 2  # Use 2 for padding
        tgt[b, :T] = seq  # predict current symbol (x_t) from past k symbols
    return toks, tgt


def collate_fn_ar(batch):
    """Collate for standard autoregressive transformer.

    Pads sequences and returns (tokens, tgt) where
        tokens: [B, T] input tokens (shifted right)
        tgt   : [B, T] target tokens (original sequence)
    """
    max_len = max(seq.size(0) for seq in batch)
    B = len(batch)
    toks = torch.full((B, max_len), 0, dtype=torch.long)
    tgt = torch.full((B, max_len), 0, dtype=torch.long)
    for b, seq in enumerate(batch):
        L = seq.size(0)
        toks[b, 1:L] = seq[:-1]
        tgt[b, :L] = seq
    return toks, tgt

# ============================================================
# Transformer model definitions
# ============================================================

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 10000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer('pe', pe)

    def forward(self, x):  # x: [B, T, d_model]
        T = x.size(1)
        return x + self.pe[:, :T]


class BinaryTransformer(nn.Module):
    """Transformer for binary‑symbol next‑token prediction.

    Supports two modes:
        * autoregressive (AR): tokens = [B, T]
        * time‑delay (TD): tokens = [B, T, k+1] representing a window of past k symbols
    """

    def __init__(self, vocab_size: int, d_model: int = 256, n_layers: int = 4, n_heads: int = 8,
                 max_delay: Optional[int] = None):
        super().__init__()
        self.td_mode = max_delay is not None
        if self.td_mode:
            self.k = max_delay
            # Embed each of k+1 positions separately, then sum
            self.embeddings = nn.ModuleList([
                nn.Embedding(vocab_size, d_model) for _ in range(self.k + 1)
            ])
        else:
            self.embed = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model, n_heads, dim_feedforward=4 * d_model,
                                                   batch_first=True)
        self.tr = nn.TransformerEncoder(encoder_layer, n_layers)
        self.proj = nn.Linear(d_model, vocab_size)

    def forward(self, tokens: torch.Tensor):
        if self.td_mode:
            # tokens: [B, T, k+1]
            B, T, K1 = tokens.shape
            embeds = 0
            for d in range(K1):
                embeds = embeds + self.embeddings[d](tokens[..., d])  # broadcast sum
        else:
            embeds = self.embed(tokens)  # [B, T, d]

        x = self.pos_encoding(embeds)
        x = self.tr(x)
        logits = self.proj(x)
        return logits  # [B, T, vocab_size]

# ============================================================
# Training loop
# ============================================================

def train(model: nn.Module, loader: DataLoader, dev_loader: DataLoader, epochs: int, lr: float,
          device: torch.device, out_dir: Path):
    optim = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.98))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=epochs * len(loader))
    loss_fn = nn.CrossEntropyLoss(ignore_index=2)  # Ignore padding token (2)
    best_ppl = float('inf')
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Starting training for {epochs} epochs...")
    print(f"Train batches per epoch: {len(loader)}")
    print(f"Dev batches: {len(dev_loader)}")
    print(f"Learning rate: {lr}")
    print(f"Device: {device}")
    print("-" * 70)

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss, steps = 0.0, 0
        train_correct, train_total = 0, 0
        
        # Progress tracking
        log_interval = max(1, len(loader) // 10)  # Log 10 times per epoch
        
        for batch_idx, (toks, tgt) in enumerate(loader):
            toks, tgt = toks.to(device), tgt.to(device)
            logits = model(toks)
            loss = loss_fn(logits.view(-1, logits.size(-1)), tgt.view(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step(); sched.step(); optim.zero_grad()
            total_loss += loss.item(); steps += 1
            
            # Calculate training accuracy for this batch
            with torch.no_grad():
                predictions = torch.argmax(logits, dim=-1)
                mask = (tgt != 2)  # Non-padding positions
                correct = (predictions == tgt) & mask
                train_correct += correct.sum().item()
                train_total += mask.sum().item()
            
            # Log progress within epoch
            if batch_idx % log_interval == 0:
                current_lr = sched.get_last_lr()[0]
                batch_acc = correct.sum().item() / mask.sum().item() if mask.sum().item() > 0 else 0.0
                print(f"  Epoch {epoch:02d} [{batch_idx:4d}/{len(loader):4d}] "
                      f"Loss: {loss.item():.4f} Acc: {batch_acc:.3f} LR: {current_lr:.6f}")
        
        # Calculate epoch metrics
        train_ppl = math.exp(total_loss / steps)
        train_acc = train_correct / train_total if train_total > 0 else 0.0
        
        # Evaluate on dev set
        dev_metrics = evaluate(model, dev_loader, device, loss_fn)
        
        print(f"Epoch {epoch:02d}: "
              f"train ppl={train_ppl:.3f} train acc={train_acc:.3f} | "
              f"dev ppl={dev_metrics['perplexity']:.3f} dev acc={dev_metrics['accuracy']:.3f}")
        print(f"  Token accuracies - 0: {dev_metrics['token_0_acc']:.3f} "
              f"(n={dev_metrics['token_0_count']:,}) | "
              f"1: {dev_metrics['token_1_acc']:.3f} "
              f"(n={dev_metrics['token_1_count']:,})")
        
        # Show prediction examples for first and last epoch
        if epoch == 1 or epoch == epochs:
            show_predictions(model, dev_loader, device)
        
        if dev_metrics['perplexity'] < best_ppl:
            best_ppl = dev_metrics['perplexity']
            torch.save(model.state_dict(), out_dir / 'best_model.pt')
            print(f"  -> New best model saved! (dev ppl: {best_ppl:.3f})")

    print("-" * 70)
    print(f"Training complete. Best dev perplexity: {best_ppl:.3f}")


@torch.inference_mode()
def show_predictions(model: nn.Module, loader: DataLoader, device: torch.device, num_examples: int = 3):
    """Show some example predictions for debugging."""
    model.eval()
    examples_shown = 0
    
    print("\nSample predictions:")
    print("-" * 50)
    
    for toks, tgt in loader:
        if examples_shown >= num_examples:
            break
            
        toks, tgt = toks.to(device), tgt.to(device)
        logits = model(toks)
        predictions = torch.argmax(logits, dim=-1)
        
        # Show first sequence in batch
        seq_len = (tgt[0] != 2).sum().item()  # Find actual sequence length
        if seq_len > 0:
            target_seq = tgt[0][:seq_len].cpu().numpy()
            pred_seq = predictions[0][:seq_len].cpu().numpy()
            
            print(f"Example {examples_shown + 1}:")
            print(f"  Target:     {''.join(map(str, target_seq))}")
            print(f"  Prediction: {''.join(map(str, pred_seq))}")
            print(f"  Accuracy:   {(target_seq == pred_seq).mean():.3f}")
            print()
            
            examples_shown += 1


@torch.inference_mode()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device, loss_fn):
    model.eval()
    total_loss, steps = 0.0, 0
    correct_predictions = 0
    total_predictions = 0
    token_counts = {0: 0, 1: 0}  # Count of each token type
    token_correct = {0: 0, 1: 0}  # Correct predictions per token type
    
    for toks, tgt in loader:
        toks, tgt = toks.to(device), tgt.to(device)
        logits = model(toks)
        loss = loss_fn(logits.view(-1, logits.size(-1)), tgt.view(-1))
        total_loss += loss.item()
        steps += 1
        
        # Calculate accuracy (ignore padding tokens)
        predictions = torch.argmax(logits, dim=-1)
        mask = (tgt != 2)  # Non-padding positions
        correct = (predictions == tgt) & mask
        correct_predictions += correct.sum().item()
        total_predictions += mask.sum().item()
        
        # Per-token accuracy
        for token_id in [0, 1]:
            token_mask = (tgt == token_id)
            token_counts[token_id] += token_mask.sum().item()
            token_correct[token_id] += (correct & token_mask).sum().item()
    
    # Calculate metrics
    accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0.0
    perplexity = math.exp(total_loss / steps)
    
    # Per-token accuracies
    token_accuracies = {}
    for token_id in [0, 1]:
        if token_counts[token_id] > 0:
            token_accuracies[token_id] = token_correct[token_id] / token_counts[token_id]
        else:
            token_accuracies[token_id] = 0.0
    
    return {
        'perplexity': perplexity,
        'accuracy': accuracy,
        'token_0_acc': token_accuracies[0],
        'token_1_acc': token_accuracies[1],
        'token_0_count': token_counts[0],
        'token_1_count': token_counts[1],
        'total_tokens': total_predictions
    }

# ============================================================
# Entry‑point
# ============================================================


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--train', type=Path, required=True, help='Path to .pt training file')
    p.add_argument('--dev', type=Path, required=True, help='Path to .pt dev file')
    p.add_argument('--out', type=Path, default=Path('checkpoints'))
    p.add_argument('--mode', choices=['ar', 'td'], default='ar')
    p.add_argument('--delay', type=int, default=10, help='k for time‑delay mode')
    p.add_argument('--d_model', type=int, default=256)
    p.add_argument('--layers', type=int, default=4)
    p.add_argument('--heads', type=int, default=8)
    p.add_argument('--epochs', type=int, default=20)
    p.add_argument('--batch', type=int, default=128)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--device', type=str, default='cuda')
    args = p.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    
    print("=" * 60)
    print("TIME DELAY TRANSFORMER TRAINING")
    print("=" * 60)
    print(f"Mode: {args.mode.upper()}")
    print(f"Train file: {args.train}")
    print(f"Dev file: {args.dev}")
    print(f"Output dir: {args.out}")
    if args.mode == 'td':
        print(f"Time delay (k): {args.delay}")
    print(f"Model: d_model={args.d_model}, layers={args.layers}, heads={args.heads}")
    print(f"Training: epochs={args.epochs}, batch={args.batch}, lr={args.lr}")
    print(f"Device: {device}")
    print()

    # Dataset and DataLoader
    print("Loading datasets...")
    train_ds = SequenceDataset(args.train)
    dev_ds = SequenceDataset(args.dev)
    
    print(f"Train sequences: {len(train_ds)}")
    print(f"Dev sequences: {len(dev_ds)}")
    
    # Show some sequence length statistics
    train_lengths = [len(seq) for seq in train_ds.data[:1000]]  # Sample first 1000
    print(f"Sample train seq lengths - min: {min(train_lengths)}, max: {max(train_lengths)}, "
          f"avg: {sum(train_lengths)/len(train_lengths):.1f}")

    if args.mode == 'td':
        collate = lambda batch: collate_fn_time_delay(batch, args.delay)
    else:
        collate = collate_fn_ar

    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True, collate_fn=collate)
    dev_loader = DataLoader(dev_ds, batch_size=args.batch, shuffle=False, collate_fn=collate)

    print(f"Train batches: {len(train_loader)}")
    print(f"Dev batches: {len(dev_loader)}")
    print()

    # Model
    print("Initializing model...")
    model = BinaryTransformer(vocab_size=3, d_model=args.d_model, n_layers=args.layers,
                              n_heads=args.heads,
                              max_delay=args.delay if args.mode == 'td' else None)
    model.to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {total_params:,} total, {trainable_params:,} trainable")
    print()

    train(model, train_loader, dev_loader, args.epochs, args.lr, device, args.out)


if __name__ == '__main__':
    main()
