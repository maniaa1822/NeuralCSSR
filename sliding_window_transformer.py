import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, silhouette_score
import json
from pathlib import Path

class SlidingWindowAttention(nn.Module):
    """Attention mechanism with sliding window"""
    
    def __init__(self, d_model, num_heads, window_size=10, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.window_size = window_size
        self.head_dim = d_model // num_heads
        
        assert d_model % num_heads == 0
        
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = 1.0 / (self.head_dim ** 0.5)
    
    def forward(self, x):
        batch_size, seq_len, d_model = x.shape
        
        # Linear projections
        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        # Create sliding window mask
        mask = self.create_sliding_window_mask(seq_len, device=x.device)
        scores = scores.masked_fill(mask == 0, float('-inf'))
        
        # Apply causal mask (can't attend to future)
        causal_mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool().to(x.device)
        scores = scores.masked_fill(causal_mask, float('-inf'))
        
        # Softmax and apply attention
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        out = torch.matmul(attn_weights, v)
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)
        
        return self.out_proj(out), attn_weights
    
    def create_sliding_window_mask(self, seq_len, device):
        """Create sliding window mask - each position can only attend to last window_size positions"""
        mask = torch.zeros(seq_len, seq_len, device=device)
        
        for i in range(seq_len):
            start = max(0, i - self.window_size + 1)
            mask[i, start:i+1] = 1  # Can attend to positions [start, i]
        
        return mask

class SlidingWindowTransformerBlock(nn.Module):
    """Transformer block with sliding window attention"""
    
    def __init__(self, d_model, num_heads, window_size, ff_dim, dropout=0.1):
        super().__init__()
        self.attention = SlidingWindowAttention(d_model, num_heads, window_size, dropout)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, ff_dim),
            nn.GELU(),
            nn.Linear(ff_dim, d_model),
            nn.Dropout(dropout)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        # Self-attention with residual connection
        attn_out, attn_weights = self.attention(x)
        x = self.norm1(x + self.dropout(attn_out))
        
        # Feed-forward with residual connection
        ff_out = self.feed_forward(x)
        x = self.norm2(x + ff_out)
        
        return x, attn_weights

class SlidingWindowTransformer(nn.Module):
    """Transformer with sliding window attention for sequence modeling"""
    
    def __init__(self, vocab_size=2, d_model=64, num_heads=4, num_layers=3, 
                 window_size=8, ff_dim=128, max_seq_len=100, dropout=0.1):
        super().__init__()
        
        self.d_model = d_model
        self.window_size = window_size
        self.max_seq_len = max_seq_len
        
        # Embeddings
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_seq_len, d_model)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            SlidingWindowTransformerBlock(d_model, num_heads, window_size, ff_dim, dropout)
            for _ in range(num_layers)
        ])
        
        # Output head
        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size)
        
        self.dropout = nn.Dropout(dropout)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(self, input_ids, return_hidden_states=False):
        batch_size, seq_len = input_ids.shape
        
        # Create position indices
        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(batch_size, -1)
        
        # Embeddings
        token_embeds = self.token_embedding(input_ids)
        pos_embeds = self.position_embedding(positions)
        x = self.dropout(token_embeds + pos_embeds)
        
        # Store hidden states and attention weights
        hidden_states = []
        attention_weights = []
        
        # Pass through transformer blocks
        for block in self.blocks:
            x, attn_weights = block(x)
            hidden_states.append(x)
            attention_weights.append(attn_weights)
        
        # Final layer norm and output
        x = self.ln_f(x)
        logits = self.lm_head(x)
        
        if return_hidden_states:
            return {
                'logits': logits,
                'hidden_states': hidden_states,
                'attention_weights': attention_weights,
                'last_hidden_state': x
            }
        else:
            return {'logits': logits}
    
    def extract_context_representations(self, input_ids, layer=-1):
        """Extract hidden representations for causal state analysis"""
        with torch.no_grad():
            outputs = self.forward(input_ids, return_hidden_states=True)
            
            if layer == -1:
                return outputs['last_hidden_state']
            else:
                return outputs['hidden_states'][layer]

class FSMSequenceDataset(Dataset):
    """Dataset for autoregressive training on FSM sequences"""
    
    def __init__(self, sequences, seq_len=50):
        self.examples = []
        self.seq_len = seq_len
        
        for seq in sequences:
            # Create overlapping windows
            for i in range(len(seq) - seq_len):
                input_seq = seq[i:i+seq_len]
                target_seq = seq[i+1:i+seq_len+1]
                
                self.examples.append({
                    'input_ids': torch.tensor(input_seq, dtype=torch.long),
                    'targets': torch.tensor(target_seq, dtype=torch.long)
                })
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        return self.examples[idx]

def train_sliding_window_transformer(sequences, epochs=50, lr=1e-3, window_size=8):
    """Train transformer with sliding window attention"""
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Training sliding window transformer on {device}")
    print(f"Window size: {window_size}")
    
    # Create dataset
    dataset = FSMSequenceDataset(sequences, seq_len=30)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    # Create model
    model = SlidingWindowTransformer(
        vocab_size=2,
        d_model=64,
        num_heads=4,
        num_layers=3,
        window_size=window_size,
        ff_dim=128,
        max_seq_len=100
    ).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Training examples: {len(dataset)}")
    
    # Training loop
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        correct_predictions = 0
        total_predictions = 0
        
        for batch_idx, batch in enumerate(dataloader):
            input_ids = batch['input_ids'].to(device)
            targets = batch['targets'].to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(input_ids)
            logits = outputs['logits']
            
            # Compute loss (predict all positions)
            loss = criterion(logits.view(-1, 2), targets.view(-1))
            
            # Accuracy
            predictions = torch.argmax(logits, dim=-1)
            correct_predictions += (predictions == targets).sum().item()
            total_predictions += targets.numel()
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_loss += loss.item()
            
            if batch_idx % 100 == 0:
                print(f"Epoch {epoch+1}, Batch {batch_idx}, Loss: {loss.item():.4f}")
        
        # Epoch summary
        avg_loss = total_loss / len(dataloader)
        accuracy = correct_predictions / total_predictions
        print(f"Epoch {epoch+1}: Loss={avg_loss:.4f}, Accuracy={accuracy:.4f}")
        
        # Analyze learned representations every 10 epochs
        if epoch % 10 == 0 and epoch > 0:
            analyze_learned_states(model, sequences[:100], device)
    
    return model

def find_optimal_clusters(data, max_k=10, min_k=2):
    """Find optimal number of clusters using multiple methods"""
    
    if len(data) < max_k:
        max_k = len(data) - 1
    
    if max_k < min_k:
        return min_k
    
    # Method 1: Silhouette Analysis
    silhouette_scores = []
    inertias = []
    
    k_range = range(min_k, max_k + 1)
    
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(data)
        
        # Silhouette score (higher is better)
        sil_score = silhouette_score(data, cluster_labels)
        silhouette_scores.append(sil_score)
        
        # Inertia for elbow method
        inertias.append(kmeans.inertia_)
    
    # Find best k using silhouette score
    best_sil_idx = np.argmax(silhouette_scores)
    best_k_silhouette = k_range[best_sil_idx]
    
    # Method 2: Elbow method using rate of change
    if len(inertias) >= 3:
        # Calculate rate of change (second derivative)
        rate_changes = []
        for i in range(1, len(inertias) - 1):
            rate_change = inertias[i-1] - 2*inertias[i] + inertias[i+1]
            rate_changes.append(rate_change)
        
        # Find elbow (maximum rate of change)
        elbow_idx = np.argmax(rate_changes) + 1  # +1 because we start from index 1
        best_k_elbow = k_range[elbow_idx]
    else:
        best_k_elbow = best_k_silhouette
    
    # Method 3: Gap statistic (simplified)
    gap_stats = []
    for k in k_range:
        # Fit k-means on real data
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(data)
        real_inertia = kmeans.inertia_
        
        # Generate random data and fit k-means
        random_data = np.random.uniform(
            low=data.min(axis=0), 
            high=data.max(axis=0), 
            size=data.shape
        )
        kmeans_random = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans_random.fit(random_data)
        random_inertia = kmeans_random.inertia_
        
        # Gap statistic
        gap = np.log(random_inertia) - np.log(real_inertia)
        gap_stats.append(gap)
    
    # Find best k using gap statistic (look for plateau)
    if len(gap_stats) >= 2:
        gap_diffs = np.diff(gap_stats)
        # Find where gap stops increasing significantly
        threshold = np.std(gap_diffs) * 0.5
        plateau_idx = None
        for i, diff in enumerate(gap_diffs):
            if diff < threshold:
                plateau_idx = i
                break
        
        if plateau_idx is not None:
            best_k_gap = k_range[plateau_idx]
        else:
            best_k_gap = k_range[np.argmax(gap_stats)]
    else:
        best_k_gap = best_k_silhouette
    
    # Print method results
    print(f"  Silhouette method: k={best_k_silhouette} (score={silhouette_scores[best_sil_idx]:.3f})")
    print(f"  Elbow method: k={best_k_elbow}")
    print(f"  Gap statistic: k={best_k_gap}")
    
    # Ensemble decision: use silhouette as primary, but consider consensus
    method_votes = [best_k_silhouette, best_k_elbow, best_k_gap]
    vote_counts = {k: method_votes.count(k) for k in set(method_votes)}
    
    # If there's consensus (2+ methods agree), use that
    consensus_k = None
    for k, votes in vote_counts.items():
        if votes >= 2:
            consensus_k = k
            break
    
    if consensus_k:
        optimal_k = consensus_k
        print(f"  Consensus: k={optimal_k} ({vote_counts[optimal_k]} methods agree)")
    else:
        # No consensus, use silhouette (most reliable for our case)
        optimal_k = best_k_silhouette
        print(f"  No consensus, using silhouette: k={optimal_k}")
    
    return optimal_k

def analyze_learned_states(model, sequences, device):
    """Analyze what causal states the model has learned"""
    
    # Import here to avoid circular imports
    import sys
    import os
    sys.path.append('fsm_transformer')
    try:
        from epsilon_machine import EpsilonMachine
    except ImportError:
        print("Warning: Could not import EpsilonMachine, skipping true state analysis")
        return None, None, None
    
    print(f"\n🔍 Analyzing learned representations...")
    
    model.eval()
    epsilon_machine = EpsilonMachine()
    
    # Extract representations for different history patterns
    representations = []
    true_states = []
    histories = []
    
    with torch.no_grad():
        for seq in sequences[:50]:  # Analyze subset
            for i in range(3, min(20, len(seq))):  # Start from position 3
                history = seq[max(0, i-10):i]  # Last 10 tokens as context
                # Convert int sequence back to strings for epsilon machine
                history_for_state = [str(x) for x in seq[max(0, i-3):i]]  # Last 3 for state
                true_state = epsilon_machine.get_causal_state(history_for_state)
                
                # Get model representation
                input_tensor = torch.tensor([history], device=device)
                repr_tensor = model.extract_context_representations(input_tensor)
                representation = repr_tensor[0, -1, :].cpu().numpy()  # Last position
                
                representations.append(representation)
                true_states.append(true_state)
                histories.append(history[-3:] if len(history) >= 3 else history)
    
    representations = np.array(representations)
    
    # Automatically determine optimal number of clusters
    optimal_k = find_optimal_clusters(representations)
    print(f"Optimal number of clusters detected: {optimal_k}")
    
    # Cluster representations with optimal k
    kmeans = KMeans(n_clusters=optimal_k, random_state=42)
    predicted_clusters = kmeans.fit_predict(representations)
    
    # Evaluate clustering quality
    state_to_int = {'A': 0, 'B': 1, 'C': 2}
    true_labels = [state_to_int[state] for state in true_states]
    
    ari = adjusted_rand_score(true_labels, predicted_clusters)
    print(f"Clustering quality (ARI): {ari:.3f}")
    
    # Analyze cluster composition
    print("\nCluster composition:")
    for cluster_id in range(optimal_k):
        cluster_mask = predicted_clusters == cluster_id
        cluster_states = [true_states[i] for i in range(len(true_states)) if cluster_mask[i]]
        cluster_histories = [histories[i] for i in range(len(histories)) if cluster_mask[i]]
        
        state_counts = {state: cluster_states.count(state) for state in ['A', 'B', 'C']}
        total = len(cluster_states)
        
        print(f"  Cluster {cluster_id} ({total} examples):")
        for state, count in state_counts.items():
            if count > 0:
                print(f"    State {state}: {count} ({count/total:.1%})")
        
        # Show example histories
        example_histories = cluster_histories[:3]
        print(f"    Example histories: {example_histories}")
    
    return representations, true_states, predicted_clusters

def visualize_attention_patterns(model, sequence, device):
    """Visualize what the model pays attention to"""
    
    model.eval()
    
    with torch.no_grad():
        input_tensor = torch.tensor([sequence[:20]], device=device)  # First 20 tokens
        outputs = model(input_tensor, return_hidden_states=True)
        
        # Get attention weights from last layer
        attention_weights = outputs['attention_weights'][-1]  # Last layer
        avg_attention = attention_weights[0].mean(dim=0).cpu().numpy()  # Average over heads
        
        # Plot attention pattern
        plt.figure(figsize=(10, 8))
        sns.heatmap(avg_attention, cmap='Blues', cbar=True)
        plt.xlabel('Key Position')
        plt.ylabel('Query Position') 
        plt.title(f'Attention Pattern (Window Size: {model.window_size})')
        plt.tight_layout()
        plt.savefig('attention_pattern.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return avg_attention

def main():
    # Load more predictable FSM dataset
    print("Loading more predictable FSM dataset...")
    with open('data/predictable_fsm/raw_data.json', 'r') as f:
        raw_data = json.load(f)
    
    sequences = raw_data['sequences']
    
    # Convert to integers
    int_sequences = [[int(token) for token in seq] for seq in sequences]
    
    print(f"Loaded {len(int_sequences)} sequences")
    print(f"Sample sequence: {int_sequences[0][:20]}")
    
    # Train model with window size 2 (matches true causal structure)
    window_size = 2
    print(f"\n{'='*50}")
    print(f"Training with window size: {window_size}")
    print(f"{'='*50}")
    
    model = train_sliding_window_transformer(
        int_sequences[:500],  # Use subset for faster training
        epochs=30,
        lr=1e-3,
        window_size=window_size
    )
    
    # Final analysis
    print(f"\nFinal analysis for window size {window_size}:")
    analyze_learned_states(model, int_sequences[:100], 'cuda' if torch.cuda.is_available() else 'cpu')
    
    # Visualize attention for the sequence
    visualize_attention_patterns(model, int_sequences[0], 'cuda' if torch.cuda.is_available() else 'cpu')

if __name__ == "__main__":
    main()