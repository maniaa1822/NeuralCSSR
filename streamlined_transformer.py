import argparse
import math
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader

class SequenceDataset(Dataset):
    """Loads binary symbol sequences from .dat files."""
    
    def __init__(self, file_path: Path, chunk_size: int = 1000):
        self.file_path = file_path
        self.sequences = self._load_dat_file(file_path, chunk_size)
    
    def _load_dat_file(self, dat_file: Path, chunk_size: int):
        """Load .dat file and split into chunks."""
        with open(dat_file, 'r') as f:
            content = f.read().strip()
        
        # Remove whitespace and convert to integers
        symbols = [int(c) for c in content if c in '01']
        
        # Split into non-overlapping chunks (match original behavior)
        sequences = []
        for i in range(0, len(symbols), chunk_size):
            chunk = symbols[i:i + chunk_size]
            if len(chunk) == chunk_size:  # Only add chunks of exact size
                sequences.append(torch.tensor(chunk, dtype=torch.long))
        
        print(f"Loaded .dat file: {len(symbols):,} symbols → {len(sequences)} chunks of size ~{chunk_size}")
        return sequences
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return self.sequences[idx]

def collate_fn_ar(batch):
    """Collate function for autoregressive training."""
    # Stack sequences: [B, T]
    sequences = torch.stack(batch)  # [B, T]
    
    # Create input and target
    input_ids = sequences[:, :-1]  # [B, T-1] 
    target_ids = sequences[:, 1:]  # [B, T-1]
    
    return input_ids, target_ids

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 10000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)  # [max_len, 1, d_model]
        self.register_buffer('pe', pe)

    def forward(self, x):  # x: [B, T, d_model]
        return x + self.pe[:x.size(1), :].transpose(0, 1)  # [B, T, d_model]

class BinaryTransformer(nn.Module):
    def __init__(self, vocab_size: int = 3, d_model: int = 256, n_layers: int = 4, n_heads: int = 8):
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=4 * d_model,
            dropout=0.1,
            activation='relu',
            batch_first=True
        )
        self.transformer = nn.TransformerDecoder(decoder_layer, num_layers=n_layers)
        self.output_projection = nn.Linear(d_model, vocab_size)
        
        # Initialize parameters
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, tokens: torch.Tensor):
        # tokens: [B, T]
        B, T = tokens.shape
        
        # Embedding + positional encoding
        x = self.embedding(tokens) * math.sqrt(self.d_model)  # [B, T, d_model]
        x = self.pos_encoder(x)
        
        # Create causal mask
        mask = torch.triu(torch.ones(T, T, device=tokens.device), diagonal=1).bool()
        
        # Transformer (decoder mode: tgt=x, memory=x for self-attention only)
        x = self.transformer(x, x, tgt_mask=mask)  # [B, T, d_model]
        
        # Output projection
        logits = self.output_projection(x)  # [B, T, vocab_size]
        
        return logits

def train_epoch(model, loader, optimizer, scheduler, device, loss_fn):
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
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        
        # Statistics
        total_loss += loss.item()
        predictions = logits.argmax(dim=-1)
        total_correct += (predictions == target_ids).sum().item()
        total_tokens += target_ids.numel()
        
        # Logging
        if batch_idx % 8 == 0:
            acc = total_correct / total_tokens if total_tokens > 0 else 0
            lr = optimizer.param_groups[0]['lr']
            print(f"  Epoch {batch_idx:2d} [{batch_idx:4d}/{len(loader):4d}] "
                  f"Loss: {loss.item():.4f} Acc: {acc:.3f} LR: {lr:.6f}")
    
    avg_loss = total_loss / len(loader)
    accuracy = total_correct / total_tokens if total_tokens > 0 else 0
    perplexity = math.exp(avg_loss)
    
    return avg_loss, accuracy, perplexity

def evaluate_model(model, loader, device, loss_fn):
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

def main():
    parser = argparse.ArgumentParser(description='Streamlined binary sequence transformer')
    parser.add_argument('--train', type=Path, required=True, help='Training .dat file')
    parser.add_argument('--dev', type=Path, help='Dev .dat file (optional)')
    parser.add_argument('--out', type=Path, default=Path('checkpoints'), help='Output directory')
    parser.add_argument('--d_model', type=int, default=64, help='Model dimension')
    parser.add_argument('--layers', type=int, default=2, help='Number of layers')
    parser.add_argument('--heads', type=int, default=4, help='Number of attention heads')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--batch', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--chunk-size', type=int, default=25, help='Sequence chunk size')
    parser.add_argument('--dev-split', type=float, default=0.2, help='Dev split ratio')
    
    args = parser.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("=" * 60)
    print("STREAMLINED TRANSFORMER TRAINING")
    print("=" * 60)
    print(f"Train file: {args.train}")
    print(f"Dev file: {args.dev}")
    print(f"Output dir: {args.out}")
    print(f"Model: d_model={args.d_model}, layers={args.layers}, heads={args.heads}")
    print(f"Training: epochs={args.epochs}, batch={args.batch}, lr={args.lr}")
    print(f"Device: {device}")
    print()
    
    # Create output directory
    args.out.mkdir(parents=True, exist_ok=True)
    
    # Load dataset
    print("Loading datasets...")
    dataset = SequenceDataset(args.train, chunk_size=args.chunk_size)
    
    # Split dataset if no dev file provided
    if args.dev is None:
        print(f"Single .dat file mode - splitting {args.train} into train/dev ({1-args.dev_split:.1%}/{args.dev_split:.1%})")
        train_size = int(len(dataset) * (1 - args.dev_split))
        dev_size = len(dataset) - train_size
        train_dataset, dev_dataset = torch.utils.data.random_split(dataset, [train_size, dev_size])
    else:
        train_dataset = dataset
        dev_dataset = SequenceDataset(args.dev, chunk_size=args.chunk_size)
    
    print(f"Split: {len(train_dataset)} train sequences, {len(dev_dataset)} dev sequences")
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch, shuffle=True, collate_fn=collate_fn_ar)
    dev_loader = DataLoader(dev_dataset, batch_size=args.batch, shuffle=False, collate_fn=collate_fn_ar)
    
    print(f"Train batches: {len(train_loader)}")
    print(f"Dev batches: {len(dev_loader)}")
    print()
    
    # Initialize model
    model = BinaryTransformer(
        vocab_size=3,  # 0, 1, 2 (padding)
        d_model=args.d_model,
        n_layers=args.layers,
        n_heads=args.heads
    )
    model.to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {total_params:,} total, {trainable_params:,} trainable")
    print()
    
    # Training setup - match original implementation
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.98))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs * len(train_loader))
    loss_fn = nn.CrossEntropyLoss(ignore_index=2)  # Ignore padding tokens
    
    # Training loop
    print(f"Starting training for {args.epochs} epochs...")
    print(f"Train batches per epoch: {len(train_loader)}")
    print(f"Dev batches: {len(dev_loader)}")
    print(f"Learning rate: {args.lr}")
    print(f"Device: {device}")
    print("-" * 70)
    
    best_dev_ppl = float('inf')
    
    for epoch in range(1, args.epochs + 1):
        # Training
        train_loss, train_acc, train_ppl = train_epoch(model, train_loader, optimizer, scheduler, device, loss_fn)
        
        # Evaluation
        dev_loss, dev_acc, dev_ppl = evaluate_model(model, dev_loader, device, loss_fn)
        
        # Scheduler already updated in training loop
        
        # Print epoch results
        print(f"Epoch {epoch:02d}: train ppl={train_ppl:.3f} train acc={train_acc:.3f} | "
              f"dev ppl={dev_ppl:.3f} dev acc={dev_acc:.3f}")
        
        # Save best model
        if dev_ppl < best_dev_ppl:
            best_dev_ppl = dev_ppl
            print(f"  -> New best model saved! (dev ppl: {dev_ppl:.3f})")
            
            # Save checkpoint
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'dev_ppl': dev_ppl,
                'dev_acc': dev_acc,
                'config': {
                    'vocab_size': 3,
                    'd_model': args.d_model,
                    'n_layers': args.layers,
                    'n_heads': args.heads,
                }
            }
            torch.save(checkpoint, args.out / 'best_model.pt')
    
    print("-" * 70)
    print(f"Training complete. Best dev perplexity: {best_dev_ppl:.3f}")

if __name__ == '__main__':
    main()