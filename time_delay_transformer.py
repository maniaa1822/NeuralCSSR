import argparse
import math
import sys
from pathlib import Path
from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
import json
from sklearn.metrics import accuracy_score

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
    """Loads binary symbol sequences (0/1) from either:
    - .pt file containing PyTorch tensors (unified dataset format)  
    - .dat file containing raw binary strings (transCSSR format)

    Each item is a 1‑D LongTensor of shape [T].
    Optionally loads aligned probe targets for causal state prediction.
    """

    def __init__(self, file_path: Path, chunk_size: int = 1000, probe_data_path: Optional[Path] = None, 
                 sliding_window: bool = False, stride: int = None):
        """
        Args:
            file_path: Path to .pt or .dat file
            chunk_size: For .dat files, split into chunks of this size
            probe_data_path: Optional path to aligned probe data (.pt file)
            sliding_window: Whether to use sliding window chunking
            stride: Stride for sliding window (defaults to chunk_size for non-overlapping)
        """
        self.file_path = file_path
        self.probe_data_path = probe_data_path
        self.sliding_window = sliding_window
        self.stride = stride if stride is not None else chunk_size
        
        if file_path.suffix == '.dat':
            self._load_dat_file(file_path, chunk_size)
        elif file_path.suffix == '.pt':
            self._load_pt_file(file_path)
        else:
            raise ValueError(f"Unsupported file format: {file_path.suffix}. Expected .pt or .dat")
    
    def _load_dat_file(self, dat_file: Path, chunk_size: int):
        """Load binary sequence from .dat file and split into chunks."""
        with open(dat_file, 'r') as f:
            binary_string = f.read().strip()
        
        if not binary_string:
            raise ValueError(f"Empty .dat file: {dat_file}")
        
        # Validate binary content
        valid_symbols = set('01')
        file_symbols = set(binary_string)
        if not file_symbols.issubset(valid_symbols):
            raise ValueError(f"Invalid symbols in .dat file: {file_symbols - valid_symbols}")
        
        # Convert to tensor of integers: '0' -> 0, '1' -> 1
        int_sequence = [int(c) for c in binary_string]
        
        # Split into chunks (sliding window or non-overlapping)
        self.data = []
        if self.sliding_window:
            # Sliding window chunking
            for i in range(0, len(int_sequence) - chunk_size + 1, self.stride):
                chunk = int_sequence[i:i + chunk_size]
                self.data.append(torch.tensor(chunk, dtype=torch.long))
        else:
            # Non-overlapping chunking (original behavior)
            for i in range(0, len(int_sequence), chunk_size):
                chunk = int_sequence[i:i + chunk_size]
                if len(chunk) == chunk_size:  # Only add chunks of exact size
                    self.data.append(torch.tensor(chunk, dtype=torch.long))
    
        # Load probe data if provided
        if self.probe_data_path is not None:
            if self.probe_data_path.suffix == '.states':
                # New format: .states file with space-separated integers
                states_text = self.probe_data_path.read_text().strip()
                states = [int(x) for x in states_text.split()]
                
                # Create chunks that exactly match the sequence chunks
                # IMPORTANT: States represent "state before processing symbol[i]"
                # For causal state prediction, we want the transformer to predict
                # the current causal state given the history up to that point
                # So we use the states as-is (no shift needed)
                probe_chunks = []
                
                if self.sliding_window:
                    # Sliding window: align with actual chunk positions
                    # Calculate how many chunks we actually created
                    sequence_length = len(binary_string)
                    for i in range(0, sequence_length - chunk_size + 1, self.stride):
                        if i + chunk_size <= len(states):
                            chunk_states = states[i:i + chunk_size]
                            probe_chunks.append(torch.tensor(chunk_states, dtype=torch.long))
                        else:
                            print(f"Warning: States ({len(states)}) shorter than sliding window at position {i}")
                            break
                else:
                    # Non-overlapping: use original logic
                    for i, seq_chunk in enumerate(self.data):
                        start_idx = i * chunk_size
                        actual_chunk_size = len(seq_chunk)
                        end_idx = start_idx + actual_chunk_size
                        
                        if end_idx <= len(states):
                            chunk_states = states[start_idx:end_idx]
                            probe_chunks.append(torch.tensor(chunk_states, dtype=torch.long))
                        else:
                            print(f"Warning: States ({len(states)}) shorter than sequences at chunk {i}")
                            break
                
                # Only keep probe chunks that have corresponding sequence chunks
                min_chunks = min(len(self.data), len(probe_chunks))
                self.data = self.data[:min_chunks]
                probe_chunks = probe_chunks[:min_chunks]
                
                print(f"Loaded .states file: {len(states)} states → {len(probe_chunks)} aligned chunks")
                print(f"State alignment check - Seq chunk 0 length: {len(self.data[0])}, State chunk 0 length: {len(probe_chunks[0])}")
                
                self.probe_targets = probe_chunks
                
            elif self.probe_data_path.is_file():
                # Old format: direct .pt file
                probe_data = torch.load(self.probe_data_path, weights_only=False)
                probe_states = probe_data['states']  # [n_chunks, chunk_size]
                
                # Convert to list of tensors for consistency
                self.probe_targets = [probe_states[i] for i in range(len(probe_states))]
                print(f"Loaded probe targets: {probe_states.shape} → {len(self.probe_targets)} chunks")
            else:
                # Directory format: with full_probe_data.pt
                probe_file = self.probe_data_path / "full_probe_data.pt"
                if probe_file.exists():
                    probe_data = torch.load(probe_file, weights_only=False)
                    probe_states = probe_data['states']  # [n_chunks, chunk_size]
                    
                    # Convert to list of tensors for consistency  
                    self.probe_targets = [probe_states[i] for i in range(len(probe_states))]
                    print(f"Loaded probe targets: {probe_states.shape} → {len(self.probe_targets)} chunks")
                else:
                    raise ValueError(f"Probe data file not found: {probe_file}")
            
            # Ensure probe data matches our sequence chunks
            if len(self.probe_targets) != len(self.data):
                print(f"Warning: Probe data length ({len(self.probe_targets)}) != sequence chunks ({len(self.data)})")
                min_len = min(len(self.probe_targets), len(self.data))
                self.data = self.data[:min_len]
                self.probe_targets = self.probe_targets[:min_len]
        else:
            self.probe_targets = None
    
        print(f"Loaded .dat file: {len(binary_string):,} symbols → {len(self.data)} chunks of size ~{chunk_size}")
    
    def _load_pt_file(self, pt_file: Path):
        """Load sequences from .pt file (existing logic)."""
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
        
        # Initialize probe_targets for consistency
        self.probe_targets = None

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if self.probe_targets is not None:
            return self.data[idx], self.probe_targets[idx]
        else:
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


def collate_fn_ar(batch, mask_boundary_positions=0):
    """Collate for standard autoregressive transformer.

    Pads sequences and returns (tokens, tgt) or (tokens, tgt, probe_targets) where
        tokens: [B, T] input tokens (shifted right)
        tgt   : [B, T] target tokens (original sequence)
        probe_targets: [B, T] causal state labels (if probe data available)
    
    Uses token 2 as padding consistently across both AR and TD modes.
    
    Args:
        mask_boundary_positions: Number of positions at start of each chunk to mask in probe targets
    """
    # Check if batch contains probe targets
    has_probe_targets = isinstance(batch[0], tuple) and len(batch[0]) == 2
    
    if has_probe_targets:
        sequences = [item[0] for item in batch]
        probe_targets = [item[1] for item in batch]
        max_len = max(seq.size(0) for seq in sequences)
    else:
        sequences = batch
        max_len = max(seq.size(0) for seq in sequences)
    
    B = len(batch)
    toks = torch.full((B, max_len), 2, dtype=torch.long)  # Use 2 as padding consistently
    tgt = torch.full((B, max_len), 2, dtype=torch.long)   # Use 2 as padding for targets
    
    if has_probe_targets:
        probe_tgt = torch.full((B, max_len), -100, dtype=torch.long)  # Use -100 for ignored positions in CrossEntropyLoss
    
    for b, seq in enumerate(sequences):
        L = seq.size(0)
        toks[b, 1:L] = seq[:-1]
        tgt[b, :L] = seq
        
        if has_probe_targets:
            # Handle variable-length probe targets
            probe_seq = probe_targets[b]
            probe_len = len(probe_seq)
            # Only fill up to the actual probe sequence length
            probe_tgt[b, :probe_len] = probe_seq
            
            # Mask the first N positions to handle chunk boundary issues
            if mask_boundary_positions > 0:
                probe_tgt[b, :min(mask_boundary_positions, probe_len)] = -100
    
    if has_probe_targets:
        return toks, tgt, probe_tgt
    else:
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
        
        if self.td_mode:
            # Time-delay mode: use encoder (no causal masking needed)
            encoder_layer = nn.TransformerEncoderLayer(d_model, n_heads, dim_feedforward=4 * d_model,
                                                       batch_first=True)
            self.tr = nn.TransformerEncoder(encoder_layer, n_layers)
        else:
            # Autoregressive mode: use decoder with proper causal masking
            decoder_layer = nn.TransformerDecoderLayer(d_model, n_heads, dim_feedforward=4 * d_model,
                                                       batch_first=True)
            self.tr = nn.TransformerDecoder(decoder_layer, n_layers)
        
        self.proj = nn.Linear(d_model, vocab_size)

    def forward(self, tokens: torch.Tensor):
        if self.td_mode:
            # tokens: [B, T, k+1]
            _, _, K1 = tokens.shape
            embeds = 0
            for d in range(K1):
                embeds = embeds + self.embeddings[d](tokens[..., d])  # broadcast sum
        else:
            embeds = self.embed(tokens)  # [B, T, d]

        x = self.pos_encoding(embeds)
        
        if self.td_mode:
            # Time-delay mode: no masking needed
            x = self.tr(x)
        else:
            # Autoregressive mode: decoder needs both tgt and memory
            # For autoregressive generation, tgt=memory=x (self-attention)
            x = self.tr(x, memory=x)
        
        logits = self.proj(x)
        
        # Mask out padding token (2) to prevent model from predicting it
        # Set logits for padding token to very negative value
        logits[..., 2] = -1e9
        
        return logits  # [B, T, vocab_size]

    def get_hidden_states(self, tokens: torch.Tensor):
        """Extract hidden states from all layers for probing."""
        if self.td_mode:
            # tokens: [B, T, k+1]
            _, _, K1 = tokens.shape
            embeds = 0
            for d in range(K1):
                embeds = embeds + self.embeddings[d](tokens[..., d])
        else:
            embeds = self.embed(tokens)  # [B, T, d]
        
        embeds = self.pos_encoding(embeds)
        
        # Collect hidden states from all layers
        hidden_states = []
        
        if self.td_mode:
            # For encoder, we need to manually iterate through layers
            x = embeds
            for layer in self.tr.layers:
                x = layer(x)
                hidden_states.append(x.clone())
        else:
            # For decoder, we need to handle causal masking
            tgt_mask = nn.Transformer.generate_square_subsequent_mask(embeds.size(1)).to(embeds.device)
            x = embeds
            for layer in self.tr.layers:
                x = layer(x, embeds, tgt_mask=tgt_mask)
                hidden_states.append(x.clone())
        
        return hidden_states


class MLPProbe(nn.Module):
    """MLP probe for causal state prediction with non-linear layers."""
    
    def __init__(self, input_dim: int, num_classes: int = 4, hidden_dim: int = None, dropout: float = 0.1):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = input_dim // 2  # Default to half the input dimension
        
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes)
        )
        
    def forward(self, x: torch.Tensor):
        return self.layers(x)


class LinearProbe(nn.Module):
    """Linear probe for causal state prediction."""
    
    def __init__(self, input_dim: int, num_classes: int = 4):
        super().__init__()
        self.classifier = nn.Linear(input_dim, num_classes)
        
    def forward(self, x: torch.Tensor):
        return self.classifier(x)


class ConcatenatedProbe(nn.Module):
    """Probe that concatenates representations from all layers."""
    
    def __init__(self, input_dim: int, num_layers: int, num_classes: int = 4, use_mlp: bool = False):
        super().__init__()
        total_dim = input_dim * num_layers
        
        if use_mlp:
            hidden_dim = total_dim // 2
            self.classifier = nn.Sequential(
                nn.Linear(total_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_dim // 2, num_classes)
            )
        else:
            self.classifier = nn.Linear(total_dim, num_classes)
    
    def forward(self, hidden_states_list):
        # hidden_states_list: list of [batch_size, seq_len, d_model] tensors
        concatenated = torch.cat(hidden_states_list, dim=-1)  # [batch_size, seq_len, d_model * num_layers]
        return self.classifier(concatenated)


# ============================================================
# Training loop
# ============================================================

def detect_num_classes_from_states_file(probe_data_path):
    """Detect the number of unique classes directly from the .states file."""
    if probe_data_path and probe_data_path.suffix == '.states':
        try:
            states_text = probe_data_path.read_text().strip()
            states = [int(x) for x in states_text.split()]
            unique_states = set(states)
            print(f"DEBUG: Found {len(unique_states)} unique states: {sorted(unique_states)}")
            return len(unique_states)
        except Exception as e:
            print(f"DEBUG: Error reading states file: {e}")
    return 4  # Default fallback


def detect_num_classes_from_dataset(dataset):
    """Detect the number of unique classes from a dataset with probe targets."""
    all_targets = set()
    
    # Check if dataset has probe data by looking at the first item
    if len(dataset) > 0:
        sample = dataset[0]
        print(f"DEBUG: Sample has {len(sample)} elements")
        if len(sample) >= 3:  # Has probe data
            print(f"DEBUG: Found probe data, sampling {len(dataset)} items")
            # Iterate through a sample of the dataset to collect unique targets
            sample_size = min(100, len(dataset))  # Sample first 100 items
            for i in range(sample_size):
                batch = dataset[i]
                probe_targets = batch[2]  # probe targets are the 3rd element
                targets_list = probe_targets.flatten().tolist()
                all_targets.update(targets_list)
                if i < 5:  # Debug first few batches
                    print(f"DEBUG: Batch {i} targets: {set(targets_list)}")
            print(f"DEBUG: All unique targets found: {sorted(all_targets)}")
    
    if all_targets:
        return len(all_targets)
    return 4  # Default fallback


def train(model: nn.Module, loader: DataLoader, dev_loader: DataLoader, epochs: int, lr: float,
          device: torch.device, out_dir: Path, probe_data_dir: Optional[Path] = None, probe_lr: float = 1e-3,
          probe_start_epoch: int = 1, probe_freq: int = 1, use_mlp_probe: bool = False, 
          probe_only_epochs: int = 0, use_concatenated_probe: bool = False, flow_gradients_to_transformer: bool = False, args=None):
    optim = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.98))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=epochs * len(loader))
    loss_fn = nn.CrossEntropyLoss(ignore_index=2)
    best_ppl = float('inf')
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize probes if probe data is provided
    probes = None
    probe_optimizers = None
    probe_criterion = None
    
    if probe_data_dir is not None:
        print("Initializing probes for causal state prediction...")
        
        # Auto-detect number of classes from the states file first, then dataset as fallback
        num_probe_classes = detect_num_classes_from_states_file(probe_data_dir)
        if num_probe_classes == 4:  # If default fallback, try dataset detection
            num_probe_classes = detect_num_classes_from_dataset(loader.dataset)
        print(f"Detected {num_probe_classes} probe classes")
        
        # Initialize probes for each layer
        d_model = model.proj.in_features
        n_layers = len(model.tr.layers)
        
        probes = {}
        probe_optimizers = {}
        
        for layer_idx in range(n_layers):
            if use_mlp_probe:
                probes[layer_idx] = MLPProbe(d_model, num_classes=num_probe_classes, hidden_dim=d_model//2).to(device)
            else:
                probes[layer_idx] = LinearProbe(d_model, num_classes=num_probe_classes).to(device)
            probe_optimizers[layer_idx] = torch.optim.Adam(probes[layer_idx].parameters(), lr=probe_lr)
        
        # Initialize concatenated probe if requested
        concatenated_probe = None
        concatenated_optimizer = None
        if use_concatenated_probe:
            concatenated_probe = ConcatenatedProbe(d_model, n_layers, num_classes=num_probe_classes, use_mlp=use_mlp_probe).to(device)
            concatenated_optimizer = torch.optim.Adam(concatenated_probe.parameters(), lr=probe_lr)
        
        probe_criterion = nn.CrossEntropyLoss(ignore_index=-100)
        probe_type = "MLP" if use_mlp_probe else "Linear"
        concat_info = f" + Concatenated {probe_type}" if concatenated_probe else ""
        gradient_flow_info = " (Gradient Flow Mode)" if flow_gradients_to_transformer else " (Frozen Transformer)"
        print(f"Initialized {n_layers} {probe_type} probes{concat_info} with {num_probe_classes} classes and learning rate {probe_lr}{gradient_flow_info}")
        
        if probe_only_epochs > 0:
            phase_info = "probe fine-tuning" if flow_gradients_to_transformer else "probes only"
            print(f"Will train transformer for {epochs - probe_only_epochs} epochs, then {phase_info} for {probe_only_epochs} epochs")
        print()

    # If loading a checkpoint for analysis, skip transformer training.
    if args.load_checkpoint:
        print("Checkpoint loaded. Skipping main transformer training phase.")
        transformer_epochs = 0
        # Ensure probe_only_epochs is set so we enter the probe training phase
        if args.probe_only_epochs == 0:
            args.probe_only_epochs = 10 # Default to 10 epochs of probe training if not specified
    else:
        transformer_epochs = epochs - probe_only_epochs
    print(f"Starting training for {transformer_epochs} transformer epochs + {probe_only_epochs} probe-only epochs...")
    print(f"Train batches per epoch: {len(loader)}")
    print(f"Dev batches: {len(dev_loader)}")
    print(f"Learning rate: {lr}")
    print(f"Device: {device}")
    print("-" * 70)

    # Phase 1: Train transformer (and optionally probes)
    for epoch in range(1, transformer_epochs + 1):
        model.train()
        total_loss, steps = 0.0, 0
        train_correct, train_total = 0, 0
        
        log_interval = max(1, len(loader) // 10)
        
        for batch_idx, batch_data in enumerate(loader):
            # Handle both cases: with and without probe targets
            if len(batch_data) == 3:
                toks, tgt, batch_probe_targets = batch_data
                toks, tgt, batch_probe_targets = toks.to(device), tgt.to(device), batch_probe_targets.to(device)
            else:
                toks, tgt = batch_data
                toks, tgt = toks.to(device), tgt.to(device)
                batch_probe_targets = None
            
            # Train transformer
            logits = model(toks)
            loss = loss_fn(logits.view(-1, logits.size(-1)), tgt.view(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step(); sched.step(); optim.zero_grad()
            total_loss += loss.item(); steps += 1
            
            # Train probes (if enabled and conditions met)
            should_train_probes = (probes is not None and 
                                 batch_probe_targets is not None and 
                                 epoch >= probe_start_epoch and 
                                 batch_idx % probe_freq == 0)
            
            if should_train_probes:
                if flow_gradients_to_transformer:
                    # Allow gradients to flow back to transformer
                    hidden_states = model.get_hidden_states(toks)
                    
                    # Accumulate all probe losses to avoid multiple backward passes
                    total_probe_loss = 0.0
                    probe_loss_count = 0
                    
                    # Zero all optimizers
                    optim.zero_grad()
                    for probe_opt in probe_optimizers.values():
                        probe_opt.zero_grad()
                    
                    # Calculate losses for all probes (including concatenated)
                    for layer_idx, probe in probes.items():
                        if layer_idx < len(hidden_states):
                            layer_hidden = hidden_states[layer_idx]
                            layer_hidden = layer_hidden.view(-1, layer_hidden.size(-1))
                            probe_targets = batch_probe_targets.view(-1)
                            
                            # Filter out ignored positions (-100) during training
                            valid_mask = probe_targets != -100
                            if valid_mask.sum() > 0:
                                valid_hidden = layer_hidden[valid_mask]
                                valid_targets = probe_targets[valid_mask]
                                
                                probe_logits = probe(valid_hidden)
                                probe_loss = probe_criterion(probe_logits, valid_targets)
                                total_probe_loss += probe_loss
                                probe_loss_count += 1
                    
                    # Add concatenated probe loss if present
                    if concatenated_probe is not None:
                        concatenated_optimizer.zero_grad()
                        
                        # Prepare hidden states for concatenation
                        valid_hidden_states = []
                        for layer_idx in range(len(hidden_states)):
                            layer_hidden = hidden_states[layer_idx]
                            layer_hidden = layer_hidden.view(-1, layer_hidden.size(-1))
                            probe_targets = batch_probe_targets.view(-1)
                            
                            # Filter out ignored positions (-100)
                            valid_mask = probe_targets != -100
                            if valid_mask.sum() > 0:
                                valid_hidden = layer_hidden[valid_mask]
                                valid_hidden_states.append(valid_hidden)
                        
                        if len(valid_hidden_states) > 0 and all(h.size(0) > 0 for h in valid_hidden_states):
                            probe_targets = batch_probe_targets.view(-1)
                            valid_mask = probe_targets != -100
                            valid_targets = probe_targets[valid_mask]
                            
                            concat_logits = concatenated_probe(valid_hidden_states)
                            concat_loss = probe_criterion(concat_logits, valid_targets)
                            total_probe_loss += concat_loss
                            probe_loss_count += 1
                    
                    # Single backward pass for all probes
                    if probe_loss_count > 0:
                        total_probe_loss.backward()
                        
                        # Apply gradients to transformer and all probes
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                        optim.step()
                        for probe_opt in probe_optimizers.values():
                            probe_opt.step()
                        if concatenated_probe is not None:
                            concatenated_optimizer.step()
                else:
                    # Freeze transformer representations (original behavior)
                    with torch.no_grad():
                        hidden_states = model.get_hidden_states(toks)
                    
                    # Train individual layer probes with frozen transformer
                    for layer_idx, probe in probes.items():
                        if layer_idx < len(hidden_states):
                            layer_hidden = hidden_states[layer_idx]
                            layer_hidden = layer_hidden.view(-1, layer_hidden.size(-1))
                            probe_targets = batch_probe_targets.view(-1)
                            
                            # Filter out ignored positions (-100) during training
                            valid_mask = probe_targets != -100
                            if valid_mask.sum() > 0:
                                valid_hidden = layer_hidden[valid_mask]
                                valid_targets = probe_targets[valid_mask]
                                
                                probe_logits = probe(valid_hidden)
                                probe_loss = probe_criterion(probe_logits, valid_targets)
                                
                                probe_optimizers[layer_idx].zero_grad()
                                probe_loss.backward()
                                probe_optimizers[layer_idx].step()
                    
                    # Train concatenated probe (frozen transformer mode)
                    if concatenated_probe is not None:
                        # Prepare hidden states for concatenation
                        valid_hidden_states = []
                        for layer_idx in range(len(hidden_states)):
                            layer_hidden = hidden_states[layer_idx]
                            layer_hidden = layer_hidden.view(-1, layer_hidden.size(-1))
                            probe_targets = batch_probe_targets.view(-1)
                            
                            # Filter out ignored positions (-100)
                            valid_mask = probe_targets != -100
                            if valid_mask.sum() > 0:
                                valid_hidden = layer_hidden[valid_mask]
                                valid_hidden_states.append(valid_hidden)
                        
                        if len(valid_hidden_states) > 0 and all(h.size(0) > 0 for h in valid_hidden_states):
                            probe_targets = batch_probe_targets.view(-1)
                            valid_mask = probe_targets != -100
                            valid_targets = probe_targets[valid_mask]
                            
                            concat_logits = concatenated_probe(valid_hidden_states)
                            concat_loss = probe_criterion(concat_logits, valid_targets)
                            
                            concatenated_optimizer.zero_grad()
                            concat_loss.backward()
                            concatenated_optimizer.step()
            
            # Calculate training accuracy
            with torch.no_grad():
                predictions = torch.argmax(logits, dim=-1)
                mask = (tgt != 2)
                correct = (predictions == tgt) & mask
                train_correct += correct.sum().item()
                train_total += mask.sum().item()
            
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
        
        # Evaluate probes and save best model as before
        if probes is not None:
            probe_accuracies = evaluate_probes(model, probes, dev_loader, device, concatenated_probe)
            if probe_accuracies:
                # Separate individual layer accuracies from concatenated
                layer_accs = {k: v for k, v in probe_accuracies.items() if k != 'concatenated'}
                probe_acc_str = " | ".join([f"L{idx}: {acc:.3f}" for idx, acc in layer_accs.items()])
                if 'concatenated' in probe_accuracies:
                    probe_acc_str += f" | Concat: {probe_accuracies['concatenated']:.3f}"
                print(f"  Probe accuracies - {probe_acc_str}")
        
        if dev_metrics['perplexity'] < best_ppl:
            best_ppl = dev_metrics['perplexity']
            save_model_and_probes(model, probes, probe_accuracies if probes else None, out_dir)
            print(f"  -> New best model saved! (dev ppl: {best_ppl:.3f})")

    # Phase 2: Probe-only training (transformer frozen)
    if probe_only_epochs > 0 and probes is not None:
        print(f"\n{'='*50}")
        print(f"PHASE 2: PROBE-ONLY TRAINING ({probe_only_epochs} epochs)")
        print(f"{'='*50}")
        
        if not flow_gradients_to_transformer:
            model.eval()  # Freeze transformer in eval mode
        else:
            model.train()  # Keep transformer in training mode for gradient flow
        
        for epoch in range(transformer_epochs + 1, epochs + 1):
            probe_epoch = epoch - transformer_epochs
            
            # Set probes to training mode
            for probe in probes.values():
                probe.train()
            
            total_probe_losses = {idx: 0.0 for idx in probes.keys()}
            probe_steps = 0
            
            for batch_idx, batch_data in enumerate(loader):
                if len(batch_data) == 3:
                    toks, tgt, batch_probe_targets = batch_data
                    toks, batch_probe_targets = toks.to(device), batch_probe_targets.to(device)
                else:
                    continue  # Skip batches without probe targets
                
                # Get hidden states and train probes
                if flow_gradients_to_transformer:
                    # Allow gradients to flow to transformer
                    hidden_states = model.get_hidden_states(toks)
                    
                    # Accumulate all probe losses to avoid multiple backward passes
                    total_batch_probe_loss = 0.0
                    probe_loss_count = 0
                    
                    # Zero all optimizers
                    optim.zero_grad()
                    for probe_opt in probe_optimizers.values():
                        probe_opt.zero_grad()
                    
                    # Calculate losses for all individual probes
                    for layer_idx, probe in probes.items():
                        if layer_idx < len(hidden_states):
                            layer_hidden = hidden_states[layer_idx]
                            layer_hidden = layer_hidden.view(-1, layer_hidden.size(-1))
                            probe_targets = batch_probe_targets.view(-1)
                            
                            # Filter out ignored positions (-100) during training
                            valid_mask = probe_targets != -100
                            if valid_mask.sum() > 0:
                                valid_hidden = layer_hidden[valid_mask]
                                valid_targets = probe_targets[valid_mask]
                                
                                probe_logits = probe(valid_hidden)
                                probe_loss = probe_criterion(probe_logits, valid_targets)
                                total_batch_probe_loss += probe_loss
                                probe_loss_count += 1
                                
                                total_probe_losses[layer_idx] += probe_loss.item()
                    
                    # Add concatenated probe loss if present
                    if concatenated_probe is not None:
                        concatenated_optimizer.zero_grad()
                        
                        # Prepare hidden states for concatenation
                        valid_hidden_states = []
                        for layer_idx in range(len(hidden_states)):
                            layer_hidden = hidden_states[layer_idx]
                            layer_hidden = layer_hidden.view(-1, layer_hidden.size(-1))
                            probe_targets = batch_probe_targets.view(-1)
                            
                            # Filter out ignored positions (-100)
                            valid_mask = probe_targets != -100
                            if valid_mask.sum() > 0:
                                valid_hidden = layer_hidden[valid_mask]
                                valid_hidden_states.append(valid_hidden)
                        
                        if len(valid_hidden_states) > 0 and all(h.size(0) > 0 for h in valid_hidden_states):
                            probe_targets = batch_probe_targets.view(-1)
                            valid_mask = probe_targets != -100
                            valid_targets = probe_targets[valid_mask]
                            
                            concat_logits = concatenated_probe(valid_hidden_states)
                            concat_loss = probe_criterion(concat_logits, valid_targets)
                            total_batch_probe_loss += concat_loss
                            probe_loss_count += 1
                    
                    # Single backward pass for all probes
                    if probe_loss_count > 0:
                        total_batch_probe_loss.backward()
                        
                        # Apply gradients to transformer and all probes
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                        optim.step()
                        for probe_opt in probe_optimizers.values():
                            probe_opt.step()
                        if concatenated_probe is not None:
                            concatenated_optimizer.step()
                else:
                    # Freeze transformer (original behavior)
                    with torch.no_grad():
                        hidden_states = model.get_hidden_states(toks)
                    
                    # Train individual layer probes with frozen transformer
                    for layer_idx, probe in probes.items():
                        if layer_idx < len(hidden_states):
                            layer_hidden = hidden_states[layer_idx]
                            layer_hidden = layer_hidden.view(-1, layer_hidden.size(-1))
                            probe_targets = batch_probe_targets.view(-1)
                            
                            # Filter out ignored positions (-100) during training
                            valid_mask = probe_targets != -100
                            if valid_mask.sum() > 0:
                                valid_hidden = layer_hidden[valid_mask]
                                valid_targets = probe_targets[valid_mask]
                                
                                probe_logits = probe(valid_hidden)
                                probe_loss = probe_criterion(probe_logits, valid_targets)
                                
                                probe_optimizers[layer_idx].zero_grad()
                                probe_loss.backward()
                                probe_optimizers[layer_idx].step()
                                
                                total_probe_losses[layer_idx] += probe_loss.item()
                
                    # Train concatenated probe (frozen transformer mode)
                    if concatenated_probe is not None:
                        valid_hidden_states = []
                        for layer_idx in range(len(hidden_states)):
                            layer_hidden = hidden_states[layer_idx]
                            layer_hidden = layer_hidden.view(-1, layer_hidden.size(-1))
                            probe_targets = batch_probe_targets.view(-1)
                            
                            # Filter out ignored positions (-100)
                            valid_mask = probe_targets != -100
                            if valid_mask.sum() > 0:
                                valid_hidden = layer_hidden[valid_mask]
                                valid_hidden_states.append(valid_hidden)
                        
                        if len(valid_hidden_states) > 0 and all(h.size(0) > 0 for h in valid_hidden_states):
                            probe_targets = batch_probe_targets.view(-1)
                            valid_mask = probe_targets != -100
                            valid_targets = probe_targets[valid_mask]
                            
                            concat_logits = concatenated_probe(valid_hidden_states)
                            concat_loss = probe_criterion(concat_logits, valid_targets)
                            
                            concatenated_optimizer.zero_grad()
                            concat_loss.backward()
                            concatenated_optimizer.step()
                
                probe_steps += 1
            
            # Evaluate probes
            probe_accuracies = evaluate_probes(model, probes, dev_loader, device, concatenated_probe)
            
            # Print probe training progress
            probe_loss_str = " | ".join([f"L{idx}: {total_probe_losses[idx]/probe_steps:.4f}" 
                                       for idx in probes.keys()])
            layer_accs = {k: v for k, v in probe_accuracies.items() if k != 'concatenated'}
            probe_acc_str = " | ".join([f"L{idx}: {acc:.3f}" for idx, acc in layer_accs.items()])
            if 'concatenated' in probe_accuracies:
                probe_acc_str += f" | Concat: {probe_accuracies['concatenated']:.3f}"
            
            print(f"Probe Epoch {probe_epoch:02d}: "
                  f"Losses - {probe_loss_str} | "
                  f"Accuracies - {probe_acc_str}")
            
            # Save best probe state
            save_model_and_probes(model, probes, probe_accuracies, out_dir, suffix=f"_probe_epoch_{probe_epoch}")

    print("-" * 70)
    print(f"Training complete. Best dev perplexity: {best_ppl:.3f}")


def evaluate_probes(model, probes, dev_loader, device, concatenated_probe=None):
    """Evaluate all probes and return accuracies."""
    probe_accuracies = {}
    
    model.eval()
    for probe in probes.values():
        probe.eval()
    if concatenated_probe is not None:
        concatenated_probe.eval()
    
    with torch.no_grad():
        # Evaluate individual layer probes
        for layer_idx, probe in probes.items():
            layer_predictions = []
            layer_targets = []
            
            for batch_data in dev_loader:
                if len(batch_data) == 3:
                    toks, tgt, batch_probe_targets = batch_data
                    toks, batch_probe_targets = toks.to(device), batch_probe_targets.to(device)
                    
                    hidden_states = model.get_hidden_states(toks)
                    
                    if layer_idx < len(hidden_states):
                        layer_hidden = hidden_states[layer_idx].view(-1, hidden_states[layer_idx].size(-1))
                        probe_targets = batch_probe_targets.view(-1)
                        
                        # Filter out ignored positions (-100)
                        valid_mask = probe_targets != -100
                        if valid_mask.sum() > 0:
                            valid_hidden = layer_hidden[valid_mask]
                            valid_targets = probe_targets[valid_mask]
                            
                            probe_logits = probe(valid_hidden)
                            _, predicted = torch.max(probe_logits, 1)
                            
                            layer_predictions.extend(predicted.cpu().numpy())
                            layer_targets.extend(valid_targets.cpu().numpy())
            
            if len(layer_predictions) > 0:
                accuracy = accuracy_score(layer_targets, layer_predictions)
                probe_accuracies[layer_idx] = accuracy
        
        # Evaluate concatenated probe
        if concatenated_probe is not None:
            concat_predictions = []
            concat_targets = []
            
            for batch_data in dev_loader:
                if len(batch_data) == 3:
                    toks, tgt, batch_probe_targets = batch_data
                    toks, batch_probe_targets = toks.to(device), batch_probe_targets.to(device)
                    
                    hidden_states = model.get_hidden_states(toks)
                    
                    # Prepare hidden states for concatenation
                    valid_hidden_states = []
                    probe_targets = batch_probe_targets.view(-1)
                    valid_mask = probe_targets != -100
                    
                    if valid_mask.sum() > 0:
                        for layer_idx in range(len(hidden_states)):
                            layer_hidden = hidden_states[layer_idx].view(-1, hidden_states[layer_idx].size(-1))
                            valid_hidden = layer_hidden[valid_mask]
                            valid_hidden_states.append(valid_hidden)
                        
                        valid_targets = probe_targets[valid_mask]
                        
                        concat_logits = concatenated_probe(valid_hidden_states)
                        _, predicted = torch.max(concat_logits, 1)
                        
                        concat_predictions.extend(predicted.cpu().numpy())
                        concat_targets.extend(valid_targets.cpu().numpy())
            
            if len(concat_predictions) > 0:
                accuracy = accuracy_score(concat_targets, concat_predictions)
                probe_accuracies['concatenated'] = accuracy
    
    return probe_accuracies


def save_model_and_probes(model, probes, probe_accuracies, out_dir, suffix=""):
    """Save model and probe states."""
    save_dict = {
        'model_state_dict': model.state_dict(),
        'config': {
            'vocab_size': 3,
            'd_model': model.proj.in_features,
            'n_layers': len(model.tr.layers),
            'n_heads': model.tr.layers[0].self_attn.num_heads if hasattr(model.tr.layers[0], 'self_attn') else 1,
            'max_delay': getattr(model, 'max_delay', None)
        }
    }
    
    if probes is not None:
        save_dict['probe_state_dicts'] = {k: v.state_dict() for k, v in probes.items()}
        if probe_accuracies:
            save_dict['probe_accuracies'] = probe_accuracies
    
    filename = f'best_model{suffix}.pt'
    torch.save(save_dict, out_dir / filename)


# ============================================================
# Evaluation
# ============================================================

@torch.inference_mode()
def show_predictions(model: nn.Module, loader: DataLoader, device: torch.device, num_examples: int = 3):
    """Show some example predictions for debugging."""
    model.eval()
    examples_shown = 0
    
    print("\nSample predictions:")
    print("-" * 50)
    
    for batch_data in loader:
        if examples_shown >= num_examples:
            break
        
        # Handle both cases: with and without probe targets
        if len(batch_data) == 3:  # (toks, tgt, probe_targets)
            toks, tgt, _ = batch_data  # Ignore probe targets in show_predictions
        else:  # (toks, tgt)
            toks, tgt = batch_data
            
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
    
    for batch_data in loader:
        # Handle both cases: with and without probe targets
        if len(batch_data) == 3:  # (toks, tgt, probe_targets)
            toks, tgt, _ = batch_data  # Ignore probe targets in evaluation
        else:  # (toks, tgt)
            toks, tgt = batch_data
            
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
    p = argparse.ArgumentParser(description='Train transformer on binary sequences (.pt or .dat files)')
    p.add_argument('--train', type=Path, help='Path to training file (.pt or .dat)')
    p.add_argument('--dev', type=Path, help='Path to dev file (.pt or .dat). If not provided for .dat, will use 80/20 train/dev split')
    p.add_argument('--out', type=Path, default=Path('checkpoints'))
    p.add_argument('--mode', choices=['ar', 'td'], default='ar', help='Autoregressive (ar) or time-delay (td) mode')
    p.add_argument('--delay', type=int, default=10, help='k for time‑delay mode')
    p.add_argument('--chunk-size', type=int, default=1000, help='Chunk size for splitting .dat files into sequences')
    p.add_argument('--dev-split', type=float, default=0.2, help='Dev split ratio when using single .dat file')
    p.add_argument('--d_model', type=int, default=256)
    p.add_argument('--layers', type=int, default=4)
    p.add_argument('--heads', type=int, default=8)
    p.add_argument('--epochs', type=int, default=20)
    p.add_argument('--batch', type=int, default=128)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--device', type=str, default='cuda')
    # Probe arguments
    p.add_argument('--probe-data', type=Path, help='Path to probe data directory (enables probe training)')
    p.add_argument('--probe-lr', type=float, default=1e-3, help='Learning rate for probes')
    p.add_argument('--probe-start-epoch', type=int, default=1, help='Start probe training from this epoch')
    p.add_argument('--probe-freq', type=int, default=1, help='Train probes every N batches (1=every batch)')
    p.add_argument('--use-mlp-probe', action='store_true', help='Use MLP probe instead of linear probe')
    p.add_argument('--probe-only-epochs', type=int, default=0, help='Number of epochs to train only probes (transformer frozen)')
    p.add_argument('--mask-boundary-positions', type=int, default=0, help='Mask first N positions of each chunk in probe training')
    p.add_argument('--sliding-window', action='store_true', help='Use sliding window chunking instead of non-overlapping')
    p.add_argument('--stride', type=int, help='Stride for sliding window (defaults to chunk_size/2 if sliding window enabled)')
    p.add_argument('--use-concatenated-probe', action='store_true', help='Use concatenated multi-layer probe in addition to individual layer probes')
    p.add_argument('--flow-gradients-to-transformer', action='store_true', help='Allow probe gradients to flow back to transformer (fine-tuning mode)')
    p.add_argument('--load-checkpoint', type=Path, help='Load a pre-trained model checkpoint to run probe-only analysis')
    p.add_argument('--eval-only', type=Path, help='Evaluate a loaded model on a specific .dat file and exit')

    args = p.parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    # Handle eval-only mode first
    if args.eval_only:
        if not args.load_checkpoint:
            raise ValueError("--load-checkpoint must be specified when using --eval-only")
        print("--- ZERO-SHOT EVALUATION MODE ---")
        print(f"Loading model from checkpoint: {args.load_checkpoint}")
        checkpoint = torch.load(args.load_checkpoint, map_location=device)
        config = checkpoint['config']
        model = BinaryTransformer(
            vocab_size=config['vocab_size'],
            d_model=config['d_model'],
            n_layers=config['n_layers'],
            n_heads=config['n_heads'],
            max_delay=config.get('max_delay')
        )
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        model.eval()
        print("Model loaded successfully.")
        print(f"Loading evaluation dataset: {args.eval_only}")
        eval_ds = SequenceDataset(args.eval_only, chunk_size=args.chunk_size)
        collate = lambda batch: collate_fn_ar(batch, 0)
        eval_loader = DataLoader(eval_ds, batch_size=args.batch, shuffle=False, collate_fn=collate)
        loss_fn = nn.CrossEntropyLoss(ignore_index=2)
        eval_metrics = evaluate(model, eval_loader, device, loss_fn)

        print("\nEvaluation Complete:")
        print(f"  Accuracy:   {eval_metrics['accuracy']:.4f}")
        print(f"  Perplexity: {eval_metrics['perplexity']:.4f}")
        return  # Exit after evaluation

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
    
    # Determine probe data paths if provided
    if args.probe_data:
        if args.probe_data.suffix == '.states':
            # Direct .states file - use for both train and dev (splitting handled in dataset)
            train_probe_path = args.probe_data
            val_probe_path = args.probe_data
        elif args.probe_data.is_file():
            # Direct .pt file - use for both train and dev
            train_probe_path = args.probe_data
            val_probe_path = args.probe_data
        else:
            # Directory with probe data files
            full_probe_path = args.probe_data / 'full_probe_data.pt'
            if full_probe_path.exists():
                # Use full probe data for both train and dev (let dataset splitting handle it)
                train_probe_path = full_probe_path
                val_probe_path = full_probe_path
            else:
                # Use separate train/val probe data
                train_probe_path = args.probe_data / 'train_probe_data.pt'
                val_probe_path = args.probe_data / 'val_probe_data.pt'
    else:
        train_probe_path = None
        val_probe_path = None
    
    # Set up sliding window parameters
    sliding_window = args.sliding_window
    stride = args.stride
    if sliding_window and stride is None:
        stride = args.chunk_size // 2  # Default to 50% overlap
    
    if args.train and args.train.suffix == '.dat' and args.dev is None:
        # Single .dat file - handle splitting differently for sliding windows
        if sliding_window:
            print(f"Single .dat file with sliding window - temporal split ({1-args.dev_split:.1%}/{args.dev_split:.1%})")
            print(f"Using chunk_size={args.chunk_size}, stride={stride}")
            
            # Load full sequence first to determine temporal split point
            with open(args.train, 'r') as f:
                full_sequence = f.read().strip()
            
            # Temporal split: first 80% for train, last 20% for dev
            split_point = int(len(full_sequence) * (1 - args.dev_split))
            
            # Create temporary files for train/dev portions
            train_seq = full_sequence[:split_point]
            dev_seq = full_sequence[split_point - args.chunk_size:]  # Overlap to ensure dev chunks have full context
            
            # Handle probe data temporal split
            if train_probe_path and train_probe_path.suffix == '.states':
                full_states = train_probe_path.read_text().strip().split()
                train_states = full_states[:split_point]
                dev_states = full_states[split_point - args.chunk_size:]
                
                # Create temporary state files
                import tempfile
                train_states_file = tempfile.NamedTemporaryFile(mode='w', suffix='.states', delete=False)
                dev_states_file = tempfile.NamedTemporaryFile(mode='w', suffix='.states', delete=False)
                
                train_states_file.write(' '.join(train_states))
                dev_states_file.write(' '.join(dev_states))
                train_states_file.close()
                dev_states_file.close()
                
                train_probe_path = Path(train_states_file.name)
        else:
            # Original non-sliding window logic
            print(f"Single .dat file mode - splitting {args.train} into train/dev ({1-args.dev_split:.1%}/{args.dev_split:.1%})")
            
            # Create separate datasets for train and dev with appropriate probe data
            train_ds = SequenceDataset(args.train, args.chunk_size, probe_data_path=train_probe_path)
            dev_ds = SequenceDataset(args.train, args.chunk_size, probe_data_path=val_probe_path)
            
            # Split sequences using the same indices for both datasets
            n_sequences = len(train_ds)
            n_train = int(n_sequences * (1 - args.dev_split))
            
            train_indices = list(range(n_train))
            dev_indices = list(range(n_train, n_sequences))
            
            from torch.utils.data import Subset
            train_ds = Subset(train_ds, train_indices)
            dev_ds = Subset(dev_ds, dev_indices)
            
            print(f"Split: {len(train_ds)} train sequences, {len(dev_ds)} dev sequences")
        
    else:
        # Separate train/dev files
        if args.dev is None:
            raise ValueError("--dev file required when using .pt files or separate .dat files")
        
        train_ds = SequenceDataset(args.train, args.chunk_size, probe_data_path=train_probe_path)
        dev_ds = SequenceDataset(args.dev, args.chunk_size, probe_data_path=val_probe_path)
        
        print(f"Train sequences: {len(train_ds)}")
        print(f"Dev sequences: {len(dev_ds)}")
    
    # Show some sequence length statistics
    if hasattr(train_ds, 'data'):
        sample_data = train_ds.data[:1000]  # Direct access
        train_lengths = [len(seq) for seq in sample_data]
    else:
        # Handle both cases: with and without probe targets
        sample_items = [train_ds[i] for i in range(min(1000, len(train_ds)))]
        if isinstance(sample_items[0], tuple):
            # With probe targets: (symbols, states)
            train_lengths = [len(item[0]) for item in sample_items]
        else:
            # Without probe targets: just symbols
            train_lengths = [len(item) for item in sample_items]
    
    print(f"Sample train seq lengths - min: {min(train_lengths)}, max: {max(train_lengths)}, "
          f"avg: {sum(train_lengths)/len(train_lengths):.1f}")

    if args.mode == 'td':
        collate = lambda batch: collate_fn_time_delay(batch, args.delay)
    else:
        collate = lambda batch: collate_fn_ar(batch, args.mask_boundary_positions)

    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True, collate_fn=collate)
    dev_loader = DataLoader(dev_ds, batch_size=args.batch, shuffle=False, collate_fn=collate)

    print(f"Train batches: {len(train_loader)}")
    print(f"Dev batches: {len(dev_loader)}")
    print()

    # Model
    print("Initializing model...")

    if args.load_checkpoint:
        print(f"Loading model from checkpoint: {args.load_checkpoint}")
        checkpoint = torch.load(args.load_checkpoint, map_location=device)
        config = checkpoint['config']
        model = BinaryTransformer(
            vocab_size=config['vocab_size'],
            d_model=config['d_model'],
            n_layers=config['n_layers'],
            n_heads=config['n_heads'],
            max_delay=config.get('max_delay')
        )
        model.load_state_dict(checkpoint['model_state_dict'])
        print("Model loaded successfully. Configuration will be overridden by checkpoint.")
        # Override args to match loaded model for consistency
        args.d_model, args.layers, args.heads = config['d_model'], config['n_layers'], config['n_heads']
    else:
        model = BinaryTransformer(vocab_size=3, d_model=args.d_model, n_layers=args.layers,
                                  n_heads=args.heads,
                                  max_delay=args.delay if args.mode == 'td' else None)

    model.to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {total_params:,} total, {trainable_params:,} trainable")
    print()

    if args.eval_only:
        if not args.load_checkpoint:
            raise ValueError("--load-checkpoint must be specified when using --eval-only")
        print("\n--- ZERO-SHOT EVALUATION MODE ---")
        print(f"Evaluating model {args.load_checkpoint} on dataset {args.eval_only}")
        eval_ds = SequenceDataset(args.eval_only, chunk_size=args.chunk_size)
        eval_loader = DataLoader(eval_ds, batch_size=args.batch, shuffle=False, collate_fn=collate)
        loss_fn = nn.CrossEntropyLoss(ignore_index=2)
        
        eval_metrics = evaluate(model, eval_loader, device, loss_fn)
        
        print("\nEvaluation Complete:")
        print(f"  Accuracy:   {eval_metrics['accuracy']:.4f}")
        print(f"  Perplexity: {eval_metrics['perplexity']:.4f}")
        return # Exit after evaluation

    train(model, train_loader, dev_loader, args.epochs, args.lr, device, args.out, 
          probe_data_dir=args.probe_data, probe_lr=args.probe_lr, 
          probe_start_epoch=args.probe_start_epoch, probe_freq=args.probe_freq,
          use_mlp_probe=args.use_mlp_probe, probe_only_epochs=args.probe_only_epochs,
          use_concatenated_probe=args.use_concatenated_probe, 
          flow_gradients_to_transformer=args.flow_gradients_to_transformer, args=args)


if __name__ == '__main__':
    main()
