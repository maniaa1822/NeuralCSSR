# Complete Time-Delay Neural CSSR Implementation Specification

## Background and Motivation

### The Core Problem: Marginal vs. Causal Learning

Standard autoregressive transformers can learn to predict next tokens by memorizing **marginal distributions** rather than discovering **causal state structure**. For example:

```python
# Standard transformer might learn:
P(next_token) = 0.5  # Just the overall 50/50 distribution

# Instead of the causal structure:
P(next_token | causal_state_A) = 0.8  # State A strongly prefers 1
P(next_token | causal_state_B) = 0.2  # State B strongly prefers 0
```

### What are Causal States?

**Causal states** are the minimal sufficient statistics for prediction. Two histories belong to the same causal state if and only if they have identical conditional future distributions:

```python
# These histories belong to the same causal state:
history_1 = [0, 1, 0, 1]  # P(next=0) = 0.8, P(next=1) = 0.2
history_2 = [1, 0, 1, 0, 1]  # P(next=0) = 0.8, P(next=1) = 0.2

# This history belongs to a different causal state:
history_3 = [1, 1, 0, 0]  # P(next=0) = 0.3, P(next=1) = 0.7
```

### Why Time-Delay Embeddings Solve This

**Time-delay embedding** comes from dynamical systems theory. Instead of processing raw sequences, we create **delay coordinate vectors** that capture the temporal structure:

```python
# Raw sequence: [0, 1, 0, 1, 1]
# 
# Time-delay embedding (max_delay=3):
# t=0: [PAD, PAD, PAD, 0]    # Insufficient history
# t=1: [PAD, PAD, 0, 1]      # History length 1
# t=2: [PAD, 0, 1, 0]        # History length 2  
# t=3: [0, 1, 0, 1]          # Full history length 3
# t=4: [1, 0, 1, 1]          # Full history length 3
```

**Key insight**: This embedding structure **forces** the transformer to learn causal relationships because:

1. **Identical delay patterns** must produce **identical predictions** (causal equivalence)
2. **Different causal states** have **different delay pattern signatures**
3. **Marginal learning becomes impossible** due to the structured temporal dependencies

## Architecture Specification

### 1. Time-Delay Embedding Layer

```python
class TimeDelayEmbedding(nn.Module):
    """
    Creates time-delay coordinate embeddings that force causal structure learning.
    
    Key innovation: Instead of standard token embeddings, create embeddings that
    capture temporal relationships through delay coordinates.
    """
    
    def __init__(self, vocab_config, max_delay=10):
        """
        Args:
            vocab_config: Dictionary with 'vocab_size', 'pad_token_id', etc.
            max_delay: Maximum number of previous timesteps to include
        """
        super().__init__()
        self.max_delay = max_delay
        self.vocab_size = vocab_config['vocab_size']  # e.g., 4 for {0, 1, PAD, UNK}
        self.pad_token_id = vocab_config['pad_token_id']  # e.g., 2
        
        # CRITICAL: Separate embedding for each delay position
        # This allows the model to learn position-specific temporal patterns
        d_model = 256  # Can be configurable
        embed_dim_per_delay = d_model // (max_delay + 1)
        
        self.delay_embeddings = nn.ModuleList([
            nn.Embedding(self.vocab_size, embed_dim_per_delay)
            for _ in range(max_delay + 1)  # [t-max_delay, ..., t-1, t]
        ])
        
        # Optional: Temporal position encoding to emphasize recency
        self.temporal_positions = nn.Parameter(
            torch.randn(max_delay + 1, embed_dim_per_delay) * 0.1
        )
        
    def forward(self, sequences):
        """
        Convert sequences to time-delay embeddings.
        
        Args:
            sequences: [batch_size, seq_len] tensor of token IDs
            
        Returns:
            delay_embeddings: [batch_size, seq_len, d_model] tensor
        """
        batch_size, seq_len = sequences.shape
        
        # For each timestep, create delay coordinate vector
        all_embeddings = []
        
        for t in range(seq_len):
            timestep_embeddings = []
            
            # Create delay vector: [x_{t-max_delay}, ..., x_{t-1}, x_t]
            for delay_idx in range(self.max_delay + 1):
                delay = self.max_delay - delay_idx  # Start from furthest delay
                
                if t - delay < 0:
                    # Use PAD token for insufficient history
                    tokens = torch.full((batch_size,), self.pad_token_id, 
                                      device=sequences.device, dtype=sequences.dtype)
                else:
                    # Use actual historical token
                    tokens = sequences[:, t - delay]
                
                # Get delay-specific embedding
                delay_emb = self.delay_embeddings[delay_idx](tokens)
                
                # Add temporal position encoding
                delay_emb = delay_emb + self.temporal_positions[delay_idx]
                
                timestep_embeddings.append(delay_emb)
            
            # Concatenate all delay embeddings for this timestep
            full_timestep_embedding = torch.cat(timestep_embeddings, dim=-1)
            all_embeddings.append(full_timestep_embedding)
        
        return torch.stack(all_embeddings, dim=1)  # [batch_size, seq_len, d_model]
```

### 2. Complete Time-Delay Neural CSSR Model

```python
class TimeDelayNeuralCSSR(nn.Module):
    """
    Complete model: Time-delay embeddings + Transformer + Prediction head.
    
    This architecture naturally discovers causal states through the delay embedding
    structure, without requiring any ground truth supervision.
    """
    
    def __init__(self, vocab_config, max_delay=10, d_model=256, n_heads=8, n_layers=4):
        super().__init__()
        self.vocab_config = vocab_config
        self.max_delay = max_delay
        self.d_model = d_model
        
        # Time-delay embedding (core innovation)
        self.delay_embedding = TimeDelayEmbedding(vocab_config, max_delay)
        
        # Standard transformer encoder (processes delay-embedded space)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=0.1,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
        # Prediction head (only predicts actual tokens, not PAD/UNK)
        # Note: vocab_config['output_vocab_size'] should be 2 for binary {0, 1}
        output_vocab_size = 2  # Only predict actual tokens {0, 1}
        self.prediction_head = nn.Linear(d_model, output_vocab_size)
        
        # Optional: Layer to extract "causal state" representations for analysis
        self.causal_state_probe = nn.Linear(d_model, 32)  # Compressed representation
        
    def forward(self, sequences, return_hidden=False):
        """
        Forward pass: sequences -> delay embeddings -> transformer -> predictions
        
        Args:
            sequences: [batch_size, seq_len] tensor of token IDs
            return_hidden: If True, also return hidden states for analysis
            
        Returns:
            logits: [batch_size, seq_len, output_vocab_size] prediction logits
            hidden_states: [batch_size, seq_len, d_model] (if return_hidden=True)
        """
        # Step 1: Create time-delay embeddings
        delay_embedded = self.delay_embedding(sequences)  # [B, L, d_model]
        
        # Step 2: Process through transformer in delay-embedded space
        # The transformer learns relationships between delay patterns
        hidden_states = self.transformer_encoder(delay_embedded)  # [B, L, d_model]
        
        # Step 3: Predict next tokens
        logits = self.prediction_head(hidden_states)  # [B, L, 2]
        
        if return_hidden:
            return logits, hidden_states
        return logits
    
    def extract_causal_state_representations(self, sequences):
        """
        Extract compressed representations that should correspond to causal states.
        
        This method can be used post-training to analyze discovered structure.
        """
        with torch.no_grad():
            delay_embedded = self.delay_embedding(sequences)
            hidden_states = self.transformer_encoder(delay_embedded)
            causal_representations = self.causal_state_probe(hidden_states)
            return causal_representations
```

## Training Specification

### 3. Training Loop

```python
def train_time_delay_neural_cssr(model, train_dataloader, val_dataloader, 
                                num_epochs=100, device='cuda'):
    """
    Simple training loop: just next-token prediction on delay-embedded sequences.
    
    The delay embedding structure automatically encourages causal state discovery
    without requiring complex loss functions.
    """
    
    # Standard setup
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    criterion = nn.CrossEntropyLoss(ignore_index=-1)  # Ignore padded positions
    
    model.to(device)
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        total_train_loss = 0
        num_batches = 0
        
        for batch in train_dataloader:
            # Extract sequences and targets from your existing data format
            sequences = batch['input_ids'].to(device)  # [batch_size, seq_len]
            targets = batch.get('target_ids', sequences[:, 1:]).to(device)  # Next tokens
            
            # Handle sequence/target alignment
            if targets.shape[1] != sequences.shape[1]:
                # If targets are shifted, align them
                sequences = sequences[:, :-1]  # Remove last token from input
            
            optimizer.zero_grad()
            
            # Forward pass through time-delay transformer
            logits = model(sequences)  # [batch_size, seq_len, 2]
            
            # Reshape for loss computation
            logits_flat = logits.reshape(-1, logits.size(-1))  # [batch_size * seq_len, 2]
            targets_flat = targets.reshape(-1)  # [batch_size * seq_len]
            
            # Compute loss (standard next-token prediction)
            loss = criterion(logits_flat, targets_flat)
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_train_loss += loss.item()
            num_batches += 1
        
        # Validation phase
        model.eval()
        total_val_loss = 0
        correct_predictions = 0
        total_predictions = 0
        
        with torch.no_grad():
            for batch in val_dataloader:
                sequences = batch['input_ids'].to(device)
                targets = batch.get('target_ids', sequences[:, 1:]).to(device)
                
                if targets.shape[1] != sequences.shape[1]:
                    sequences = sequences[:, :-1]
                
                logits = model(sequences)
                
                # Compute validation loss
                logits_flat = logits.reshape(-1, logits.size(-1))
                targets_flat = targets.reshape(-1)
                val_loss = criterion(logits_flat, targets_flat)
                
                total_val_loss += val_loss.item()
                
                # Compute accuracy
                predictions = torch.argmax(logits_flat, dim=-1)
                correct_predictions += (predictions == targets_flat).sum().item()
                total_predictions += targets_flat.numel()
        
        # Update learning rate
        scheduler.step()
        
        # Logging
        avg_train_loss = total_train_loss / num_batches
        avg_val_loss = total_val_loss / len(val_dataloader)
        accuracy = correct_predictions / total_predictions
        
        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"  Train Loss: {avg_train_loss:.4f}")
        print(f"  Val Loss: {avg_val_loss:.4f}")
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  LR: {scheduler.get_last_lr()[0]:.6f}")
        
        # Optional: Save checkpoints
        if (epoch + 1) % 20 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
                'accuracy': accuracy
            }, f'checkpoint_epoch_{epoch+1}.pt')
    
    return model
```

## Analysis and Evaluation

### 4. Causal State Discovery Analysis

```python
def analyze_discovered_causal_states(model, dataset, max_samples=5000):
    """
    Analyze the causal states discovered by the time-delay transformer.
    
    This function extracts hidden representations and analyzes their clustering
    properties to validate causal state discovery.
    """
    model.eval()
    
    # Collect data for analysis
    all_hidden_states = []
    all_delay_patterns = []
    all_predictions = []
    all_targets = []
    sample_count = 0
    
    with torch.no_grad():
        for batch in dataset:
            if sample_count >= max_samples:
                break
                
            sequences = batch['input_ids']
            targets = batch.get('target_ids', sequences[:, 1:])
            
            if targets.shape[1] != sequences.shape[1]:
                sequences = sequences[:, :-1]
            
            # Get predictions and hidden states
            logits, hidden_states = model(sequences, return_hidden=True)
            predictions = torch.argmax(logits, dim=-1)
            
            # Extract delay patterns and corresponding states
            batch_size, seq_len = sequences.shape
            
            for b in range(batch_size):
                for t in range(model.max_delay, seq_len):  # Only positions with full history
                    if sample_count >= max_samples:
                        break
                    
                    # Extract delay pattern (last max_delay+1 tokens)
                    delay_pattern = sequences[b, t-model.max_delay:t+1].cpu().numpy()
                    
                    # Extract corresponding hidden state
                    hidden_state = hidden_states[b, t].cpu().numpy()
                    
                    # Extract prediction and target
                    prediction = predictions[b, t].item()
                    target = targets[b, t].item()
                    
                    all_delay_patterns.append(delay_pattern)
                    all_hidden_states.append(hidden_state)
                    all_predictions.append(prediction)
                    all_targets.append(target)
                    sample_count += 1
    
    # Convert to numpy arrays
    delay_patterns = np.array(all_delay_patterns)
    hidden_states = np.array(all_hidden_states)
    predictions = np.array(all_predictions)
    targets = np.array(all_targets)
    
    # Cluster analysis
    analysis_results = perform_clustering_analysis(
        delay_patterns, hidden_states, predictions, targets
    )
    
    return analysis_results

def perform_clustering_analysis(delay_patterns, hidden_states, predictions, targets):
    """
    Perform comprehensive clustering analysis to validate causal state discovery.
    """
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score, adjusted_rand_score
    from sklearn.decomposition import PCA
    import matplotlib.pyplot as plt
    
    # 1. Cluster hidden states
    n_clusters_range = range(2, min(15, len(hidden_states) // 100))
    clustering_results = {}
    
    for n_clusters in n_clusters_range:
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(hidden_states)
        
        # Compute clustering quality metrics
        silhouette = silhouette_score(hidden_states, cluster_labels)
        
        clustering_results[n_clusters] = {
            'cluster_labels': cluster_labels,
            'silhouette_score': silhouette,
            'cluster_centers': kmeans.cluster_centers_
        }
    
    # Find best clustering
    best_n_clusters = max(clustering_results.keys(), 
                         key=lambda k: clustering_results[k]['silhouette_score'])
    best_clustering = clustering_results[best_n_clusters]
    
    # 2. Analyze prediction consistency within clusters
    consistency_analysis = analyze_prediction_consistency(
        best_clustering['cluster_labels'], delay_patterns, predictions, targets
    )
    
    # 3. Analyze delay pattern groupings
    pattern_analysis = analyze_delay_pattern_groupings(
        delay_patterns, best_clustering['cluster_labels'], predictions
    )
    
    # 4. Visualize clustering (PCA projection)
    pca = PCA(n_components=2)
    hidden_2d = pca.fit_transform(hidden_states)
    
    return {
        'best_n_clusters': best_n_clusters,
        'cluster_labels': best_clustering['cluster_labels'],
        'silhouette_score': best_clustering['silhouette_score'],
        'consistency_analysis': consistency_analysis,
        'pattern_analysis': pattern_analysis,
        'hidden_states_2d': hidden_2d,
        'clustering_results': clustering_results
    }

def analyze_prediction_consistency(cluster_labels, delay_patterns, predictions, targets):
    """
    Analyze whether clusters have consistent prediction behavior.
    
    This is the key test: good causal states should have low prediction entropy.
    """
    consistency_scores = []
    
    for cluster_id in np.unique(cluster_labels):
        cluster_mask = cluster_labels == cluster_id
        cluster_predictions = predictions[cluster_mask]
        cluster_targets = targets[cluster_mask]
        
        if len(cluster_predictions) > 1:
            # Calculate prediction entropy for this cluster
            unique_preds, counts = np.unique(cluster_predictions, return_counts=True)
            pred_probs = counts / counts.sum()
            pred_entropy = -np.sum(pred_probs * np.log2(pred_probs + 1e-8))
            
            # Calculate target entropy for this cluster  
            unique_targets, target_counts = np.unique(cluster_targets, return_counts=True)
            target_probs = target_counts / target_counts.sum()
            target_entropy = -np.sum(target_probs * np.log2(target_probs + 1e-8))
            
            # Consistency = 1 - normalized_entropy (higher is better)
            max_entropy = np.log2(2)  # For binary prediction
            pred_consistency = 1.0 - (pred_entropy / max_entropy)
            target_consistency = 1.0 - (target_entropy / max_entropy)
            
            consistency_scores.append({
                'cluster_id': cluster_id,
                'cluster_size': len(cluster_predictions),
                'prediction_consistency': pred_consistency,
                'target_consistency': target_consistency,
                'prediction_entropy': pred_entropy,
                'target_entropy': target_entropy
            })
    
    return consistency_scores
```

## Integration with Existing Codebase

### 5. Integration Points

**Data Loading**: Use existing dataset loaders with vocabulary config:
```python
# Load vocab config from existing dataset
dataset_path = "datasets/biased_exp"
with open(f"{dataset_path}/neural_format/dataset_info.json") as f:
    dataset_info = json.load(f)
    vocab_config = dataset_info['tokenization']

# Initialize model with existing vocabulary
model = TimeDelayNeuralCSSR(vocab_config, max_delay=10)
```

**Training Data**: Works with existing PyTorch datasets:
```python
# Load existing datasets
train_dataset = torch.load(f"{dataset_path}/neural_format/train_dataset.pt")
val_dataset = torch.load(f"{dataset_path}/neural_format/val_dataset.pt")

# Create dataloaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
```

**Evaluation**: Integrate with existing distance analysis framework:
```python
# After training, extract discovered states
discovered_states = analyze_discovered_causal_states(model, val_loader)

# Compare with ground truth using existing framework
from neural_cssr.evaluation.machine_distance import MachineDistanceCalculator
distance_analyzer = MachineDistanceCalculator()

# Convert discovered states to format compatible with distance analysis
# Then run comparison with ground truth machines
```

## Expected Outcomes

### Success Metrics

1. **Clustering Quality**: Hidden states should form discrete, well-separated clusters
2. **Prediction Consistency**: Clusters should have low prediction entropy (high consistency)
3. **Causal Correspondence**: Discovered clusters should correlate with true causal states
4. **Prediction Accuracy**: Should achieve near-optimal Bayes accuracy

### Validation Approach

1. **Train time-delay model** on generated ε-machine datasets
2. **Extract hidden state clusters** and analyze consistency
3. **Compare discovered states** with ground truth using distance metrics
4. **Validate on multiple machine types** (2-state, 3-state, biased, topological)

The time-delay embedding architecture should naturally discover causal state structure through the temporal constraints, without requiring any ground truth supervision during training.