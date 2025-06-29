import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import json
from pathlib import Path
import sys
import os

class FSMDatasetConverter:
    """Convert FSM dataset to unsupervised format (same as train_fsm_contrastive.py)"""
    
    def __init__(self, data_dir="data/fsm_transformer"):
        self.data_dir = data_dir
        
    def load_and_convert(self):
        """Load FSM dataset and convert to sequences"""
        # Load raw sequences directly
        with open(Path(self.data_dir) / 'raw_data.json', 'r') as f:
            raw_data = json.load(f)
        
        sequences = raw_data['sequences']
        state_trajectories = raw_data['state_trajectories']
        
        print(f"Loaded {len(sequences)} sequences from FSM dataset")
        
        # Convert sequences from list of strings to list of integers
        int_sequences = []
        for seq in sequences:
            int_seq = [int(token) for token in seq]
            int_sequences.append(int_seq)
        
        return int_sequences, state_trajectories

class UnsupervisedCausalStateDataset(Dataset):
    """Create training pairs without knowing true causal states"""
    
    def __init__(self, sequences, history_len=3, future_len=5):
        self.pairs = []
        self.labels = []  # 1 = similar futures, 0 = different futures
        
        # Use only 100 sequences instead of all 1000
        for seq in sequences[:100]:
            for i in range(len(seq) - history_len - future_len):
                history1 = seq[i:i+history_len]
                future1 = seq[i+history_len:i+history_len+future_len]
                
                # Create positive and negative pairs - fewer pairs per sequence
                for j in range(i+1, min(i+10, len(seq) - history_len - future_len)):
                    history2 = seq[j:j+history_len]
                    future2 = seq[j+history_len:j+history_len+future_len]
                    
                    # Similarity based on future overlap
                    future_similarity = self.compute_future_similarity(future1, future2)
                    
                    self.pairs.append((history1, history2))
                    self.labels.append(1 if future_similarity > 0.7 else 0)
    
    def compute_future_similarity(self, fut1, fut2):
        """Compute how similar two future sequences are"""
        # Simple overlap metric - could be more sophisticated
        overlap = sum(1 for a, b in zip(fut1, fut2) if a == b)
        return overlap / max(len(fut1), len(fut2))
    
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        (hist1, hist2), label = self.pairs[idx], self.labels[idx]
        return {
            'history1': torch.tensor(hist1, dtype=torch.long),
            'history2': torch.tensor(hist2, dtype=torch.long),
            'similar': torch.tensor(label, dtype=torch.float)
        }

class UnsupervisedCSSR(nn.Module):
    """Discover causal states without supervision"""
    
    def __init__(self, vocab_size=2, d_model=64, num_layers=2, max_states=8):
        super().__init__()
        
        # History encoder
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model, nhead=4, batch_first=True),
            num_layers=num_layers
        )
        
        # State discovery via vector quantization
        self.state_dim = 32
        self.max_states = max_states
        
        # Learnable state codebook
        self.state_embeddings = nn.Embedding(max_states, self.state_dim)
        self.state_embeddings.weight.data.uniform_(-1/max_states, 1/max_states)
        
        # Project to state space
        self.to_state = nn.Linear(d_model, self.state_dim)
        
        # Future predictor
        self.future_predictor = nn.Sequential(
            nn.Linear(self.state_dim, d_model),
            nn.ReLU(),
            nn.Linear(d_model, vocab_size)
        )
        
    def encode_history(self, history):
        """Encode history into continuous representation"""
        # [batch, seq_len] -> [batch, seq_len, d_model]
        embedded = self.embedding(history)
        
        # Transformer encoding
        encoded = self.transformer(embedded)
        
        # Take last position as history representation
        return encoded[:, -1, :]  # [batch, d_model]
    
    def quantize_to_state(self, continuous_repr):
        """Vector quantization to discrete states"""
        # Project to state space
        projected = self.to_state(continuous_repr)  # [batch, state_dim]
        
        # Find nearest state in codebook
        distances = torch.cdist(projected.unsqueeze(1), 
                               self.state_embeddings.weight.unsqueeze(0))  # [batch, 1, max_states]
        
        # Get closest state indices
        state_indices = torch.argmin(distances.squeeze(1), dim=1)  # [batch]
        
        # Get quantized representations
        quantized = self.state_embeddings(state_indices)  # [batch, state_dim]
        
        # Straight-through estimator for gradients
        quantized = projected + (quantized - projected).detach()
        
        # Commitment loss to encourage use of codebook
        commitment_loss = F.mse_loss(projected, quantized.detach())
        
        return quantized, state_indices, commitment_loss
    
    def forward(self, history1, history2):
        """Forward pass for contrastive learning"""
        # Encode both histories
        repr1 = self.encode_history(history1)
        repr2 = self.encode_history(history2)
        
        # Quantize to discrete states
        state1, indices1, commit_loss1 = self.quantize_to_state(repr1)
        state2, indices2, commit_loss2 = self.quantize_to_state(repr2)
        
        # Normalize for cosine similarity
        state1_norm = F.normalize(state1, p=2, dim=1)
        state2_norm = F.normalize(state2, p=2, dim=1)
        
        return {
            'state1': state1_norm,
            'state2': state2_norm,
            'indices1': indices1,
            'indices2': indices2,
            'commitment_loss': commit_loss1 + commit_loss2
        }
    
    def predict_future(self, history):
        """Predict next token from history"""
        repr = self.encode_history(history)
        state, _, _ = self.quantize_to_state(repr)
        return self.future_predictor(state)

def contrastive_loss(state1, state2, similarity_labels, temperature=0.1):
    """Contrastive loss for state learning"""
    # Compute cosine similarity
    similarity = F.cosine_similarity(state1, state2, dim=1)
    
    # Scale by temperature
    similarity = similarity / temperature
    
    # Binary classification loss
    loss = F.binary_cross_entropy_with_logits(similarity, similarity_labels)
    
    return loss

def train_unsupervised_cssr(sequences, epochs=100, lr=1e-3):
    """Train the unsupervised CSSR model"""
    
    # Device configuration
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Create dataset
    dataset = UnsupervisedCausalStateDataset(sequences)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    # Create model
    model = UnsupervisedCSSR()
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    print(f"Training on {len(dataset)} contrastive pairs")
    print(f"Number of batches per epoch: {len(dataloader)}")
    print(f"Batch size: {dataloader.batch_size}")
    print()
    
    for epoch in range(epochs):
        print(f"🚀 Starting Epoch {epoch+1}/{epochs}")
        total_loss = 0
        total_commitment = 0
        batch_count = 0
        
        for batch_idx, batch in enumerate(dataloader):
            optimizer.zero_grad()
            
            # Move batch to device
            history1 = batch['history1'].to(device)
            history2 = batch['history2'].to(device)
            similar = batch['similar'].to(device)
            
            # Forward pass
            outputs = model(history1, history2)
            
            # Contrastive loss
            contrast_loss = contrastive_loss(
                outputs['state1'], 
                outputs['state2'], 
                similar
            )
            
            # Total loss
            loss = contrast_loss + 0.25 * outputs['commitment_loss']
            
            loss.backward()
            optimizer.step()
            
            total_loss += contrast_loss.item()
            total_commitment += outputs['commitment_loss'].item()
            batch_count += 1
            
            # Log progress every 1000 batches
            if batch_idx % 1000 == 0:
                avg_loss = total_loss / (batch_count + 1e-8)
                avg_commit = total_commitment / (batch_count + 1e-8)
                print(f"  Batch {batch_idx:5d}/{len(dataloader)} | Loss: {avg_loss:.4f} | Commit: {avg_commit:.4f}")
        
        # End of epoch summary
        avg_loss = total_loss / (batch_count + 1e-8)
        avg_commit = total_commitment / (batch_count + 1e-8)
        print(f"✅ Epoch {epoch+1} Complete | Avg Loss: {avg_loss:.4f} | Avg Commit: {avg_commit:.4f}")
        print()
        
        if epoch % 10 == 0:
            print(f"📊 Analyzing discovered states at epoch {epoch+1}...")
            analyze_discovered_states(model, sequences[:100], device)
            print()
    
    return model

def analyze_discovered_states(model, sequences, device):
    """Analyze what states the model has discovered"""
    model.eval()
    
    state_counts = torch.zeros(model.max_states)
    state_examples = {i: [] for i in range(model.max_states)}
    
    with torch.no_grad():
        for seq in sequences[:100]:  # Analyze subset
            for i in range(len(seq) - 3):
                history = torch.tensor(seq[i:i+3]).unsqueeze(0).to(device)
                
                repr = model.encode_history(history)
                _, state_idx, _ = model.quantize_to_state(repr)
                
                state_idx = state_idx.item()
                state_counts[state_idx] += 1
                
                if len(state_examples[state_idx]) < 5:
                    state_examples[state_idx].append(seq[i:i+3])
    
    # Print discovered states
    active_states = (state_counts > 0).sum().item()
    print(f"\nDiscovered {active_states} active states:")
    
    for state_id in range(model.max_states):
        if state_counts[state_id] > 0:
            print(f"State {state_id}: {state_counts[state_id].item()} examples")
            print(f"  Examples: {state_examples[state_id][:3]}")
    
    return active_states, state_examples

# Example usage
if __name__ == "__main__":
    print("Loading FSM dataset (same as train_fsm_contrastive.py)...")
    
    # Load and convert FSM dataset (same format as train_fsm_contrastive.py)
    converter = FSMDatasetConverter("data/fsm_transformer")
    sequences, state_trajectories = converter.load_and_convert()
    
    print(f"Converted to {len(sequences)} sequences")
    print(f"Sample sequences: {sequences[:3]}")
    
    # Train unsupervised model
    model = train_unsupervised_cssr(sequences, epochs=50)
    
    print("\nFinal analysis:")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    analyze_discovered_states(model, sequences, device)