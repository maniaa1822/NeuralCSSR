#!/usr/bin/env python3
"""
Contrastive training script adapted for FSM transformer data.
This leverages the existing contrastive_transformer.py but uses the FSM epsilon-machine dataset.
"""

import torch
import torch.nn.functional as F
import sys
import os
import time
import json
import numpy as np
from collections import defaultdict
from pathlib import Path

# Import FSM transformer modules
from fsm_transformer.transformer import AutoregressiveTransformer
from fsm_transformer.data_generator import load_dataset
from fsm_transformer.epsilon_machine import EpsilonMachine

# Import contrastive model
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
from neural_cssr.neural.contrastive_transformer import ContrastiveNeuralCSSR, create_contrastive_model


class FSMContrastiveDataset(torch.utils.data.Dataset):
    """Convert FSM dataset to contrastive learning format."""
    
    def __init__(self, fsm_dataset, raw_sequences, state_trajectories, max_length=20):
        self.fsm_dataset = fsm_dataset
        self.raw_sequences = raw_sequences
        self.state_trajectories = state_trajectories
        self.max_length = max_length
        
        # Build contrastive examples
        self.examples = self._build_contrastive_examples()
        
    def _build_contrastive_examples(self):
        """Build examples with history -> causal state mapping."""
        examples = []
        
        epsilon_machine = EpsilonMachine()
        
        for seq_idx, (sequence, states) in enumerate(zip(self.raw_sequences[:1000], self.state_trajectories[:1000])):
            # Create multiple context windows from each sequence
            for i in range(1, min(len(sequence), self.max_length)):
                history = sequence[:i]
                causal_state = epsilon_machine.get_causal_state(history)
                
                # Convert to token IDs
                input_ids = [int(token) for token in history]
                
                # Pad to max_length
                attention_mask = [1] * len(input_ids) + [0] * (self.max_length - len(input_ids))
                input_ids = input_ids + [2] * (self.max_length - len(input_ids))  # pad_token_id = 2
                
                examples.append({
                    'input_ids': input_ids,
                    'attention_mask': attention_mask,
                    'causal_state': causal_state,
                    'history': ''.join(history),
                    'seq_idx': seq_idx
                })
                
        return examples
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        example = self.examples[idx]
        return {
            'input_ids': torch.tensor(example['input_ids'], dtype=torch.long),
            'attention_mask': torch.tensor(example['attention_mask'], dtype=torch.long),
            'causal_state': example['causal_state']
        }


def evaluate_embeddings_fsm(model, dataloader, device, num_samples=200):
    """Evaluate contrastive embeddings for FSM data."""
    model.eval()
    
    embeddings_by_state = defaultdict(list)
    
    with torch.no_grad():
        samples_collected = 0
        for batch in dataloader:
            if samples_collected >= num_samples:
                break
                
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            causal_states = batch['causal_state']
            
            # Get embeddings
            result = model(input_ids, attention_mask)
            embeddings = result['embeddings']
            
            # Check for NaN embeddings
            if torch.isnan(embeddings).any():
                continue
            
            # Collect embeddings by causal state
            for i, state in enumerate(causal_states):
                if samples_collected < num_samples:
                    embeddings_by_state[state].append(embeddings[i].cpu())
                    samples_collected += 1
    
    if len(embeddings_by_state) < 2:
        return {
            'separation_score': 0, 
            'intra_class_similarity': 0, 
            'inter_class_similarity': 0,
            'state_distribution': {}
        }
    
    # Compute metrics safely
    metrics = {}
    
    try:
        # 1. Intra-class similarity (same causal state should be similar)
        intra_similarities = []
        for state, state_embeddings in embeddings_by_state.items():
            if len(state_embeddings) > 1:
                state_tensor = torch.stack(state_embeddings)
                
                # Compute pairwise similarities
                for i in range(len(state_tensor)):
                    for j in range(i + 1, len(state_tensor)):
                        sim = F.cosine_similarity(
                            state_tensor[i:i+1], 
                            state_tensor[j:j+1], 
                            dim=1
                        ).item()
                        if not np.isnan(sim):
                            intra_similarities.append(sim)
        
        # 2. Inter-class similarity (different causal states should be dissimilar)
        inter_similarities = []
        states = list(embeddings_by_state.keys())
        for i, state1 in enumerate(states):
            for j, state2 in enumerate(states):
                if i < j:  # Avoid double counting
                    emb1_list = embeddings_by_state[state1]
                    emb2_list = embeddings_by_state[state2]
                    
                    # Sample some pairs to avoid too many computations
                    max_pairs = 5
                    for k1, emb1 in enumerate(emb1_list[:max_pairs]):
                        for k2, emb2 in enumerate(emb2_list[:max_pairs]):
                            sim = F.cosine_similarity(emb1.unsqueeze(0), emb2.unsqueeze(0), dim=1).item()
                            if not np.isnan(sim):
                                inter_similarities.append(sim)
        
        # 3. Compute metrics
        if intra_similarities:
            metrics['intra_class_similarity'] = np.mean(intra_similarities)
        else:
            metrics['intra_class_similarity'] = 0
        
        if inter_similarities:
            metrics['inter_class_similarity'] = np.mean(inter_similarities)
        else:
            metrics['inter_class_similarity'] = 0
        
        # 4. Separation score (higher is better)
        metrics['separation_score'] = metrics['intra_class_similarity'] - metrics['inter_class_similarity']
        
        # 5. State distribution
        state_counts = {state: len(embeddings) for state, embeddings in embeddings_by_state.items()}
        metrics['state_distribution'] = state_counts
        
    except Exception as e:
        print(f"  Warning: Error computing metrics: {e}")
        metrics = {
            'separation_score': 0, 
            'intra_class_similarity': 0, 
            'inter_class_similarity': 0,
            'state_distribution': {}
        }
    
    return metrics


def train_fsm_contrastive():
    """Train contrastive model on FSM epsilon-machine data."""
    
    print("🎯 Training FSM Contrastive Neural CSSR")
    print("="*50)
    
    # Configuration
    batch_size = 32
    learning_rate = 5e-5  # Lower for stability
    num_epochs = 40
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"Configuration:")
    print(f"  Batch size: {batch_size}")
    print(f"  Learning rate: {learning_rate}")
    print(f"  Epochs: {num_epochs}")
    print(f"  Device: {device}")
    
    # Load FSM dataset
    data_dir = "data/fsm_transformer"
    print(f"\nLoading FSM dataset from {data_dir}...")
    
    # Load processed dataset
    dataset, metadata = load_dataset(data_dir)
    
    # Load raw sequences for causal state mapping
    with open(Path(data_dir) / 'raw_data.json', 'r') as f:
        raw_data = json.load(f)
    
    sequences = raw_data['sequences']
    state_trajectories = raw_data['state_trajectories']
    
    print(f"Dataset loaded: {len(sequences)} sequences")
    print(f"Metadata: {len(dataset)} autoregressive examples")
    
    # Create contrastive datasets
    train_size = int(0.8 * len(sequences))
    train_sequences = sequences[:train_size]
    train_states = state_trajectories[:train_size]
    val_sequences = sequences[train_size:]
    val_states = state_trajectories[train_size:]
    
    train_contrastive_dataset = FSMContrastiveDataset(
        dataset, train_sequences, train_states, max_length=20
    )
    val_contrastive_dataset = FSMContrastiveDataset(
        dataset, val_sequences, val_states, max_length=20
    )
    
    # Create dataloaders
    train_dataloader = torch.utils.data.DataLoader(
        train_contrastive_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=0  # Avoid multiprocessing issues
    )
    
    val_dataloader = torch.utils.data.DataLoader(
        val_contrastive_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )
    
    print(f"Contrastive datasets:")
    print(f"  Train: {len(train_contrastive_dataset)} examples, {len(train_dataloader)} batches")
    print(f"  Val: {len(val_contrastive_dataset)} examples, {len(val_dataloader)} batches")
    
    # Create vocab info for model
    vocab_info = {
        'vocab_size': 2,  # Binary alphabet {0, 1}
        'pad_token_id': 2,
        'max_length': 20
    }
    
    # Create contrastive model optimized for FSM
    model = create_contrastive_model(
        vocab_info, 
        d_model=64,        # Larger for better representation
        embedding_dim=32,  # Good embedding dimension
        num_layers=2,      # Enough complexity
        num_heads=4,
        dropout=0.1,
        max_seq_length=20
    )
    model = model.to(device)
    
    # Optimizer with weight decay
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=0.01,
        eps=1e-8
    )
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=8
    )
    
    # Training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'separation_score': [],
        'intra_class_similarity': [],
        'inter_class_similarity': []
    }
    
    best_separation = -float('inf')
    patience_counter = 0
    max_patience = 15
    
    # Create output directory
    output_dir = Path("fsm_contrastive_results")
    output_dir.mkdir(exist_ok=True)
    
    print(f"\n🚀 Starting training...")
    
    for epoch in range(num_epochs):
        start_time = time.time()
        
        # Training
        model.train()
        train_loss = 0
        train_batches = 0
        
        print(f"\n📚 Training Epoch {epoch+1}/{num_epochs}...")
        
        for batch_idx, batch in enumerate(train_dataloader):
            if batch_idx % 100 == 0:
                print(f"  Batch {batch_idx}/{len(train_dataloader)} - LR: {optimizer.param_groups[0]['lr']:.2e}")
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            causal_states = batch['causal_state']
            
            optimizer.zero_grad()
            
            # Forward pass
            result = model(input_ids, attention_mask, causal_states)
            loss = result['contrastive_loss']
            
            # Check for NaN
            if torch.isnan(loss):
                print(f"  ⚠️  Warning: NaN loss in training batch {batch_idx}")
                continue
            
            loss.backward()
            
            # Gradient clipping
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            train_loss += loss.item()
            train_batches += 1
            
            # Log progress every 100 batches
            if batch_idx % 100 == 0 and batch_idx > 0:
                avg_loss = train_loss / train_batches
                print(f"    Loss: {avg_loss:.4f}, Grad norm: {grad_norm:.3f}")
        
        avg_train_loss = train_loss / train_batches if train_batches > 0 else 0
        
        # Validation
        print(f"🔍 Validating...")
        model.eval()
        val_loss = 0
        val_batches = 0
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(val_dataloader):
                if batch_idx % 50 == 0:
                    print(f"  Val batch {batch_idx}/{len(val_dataloader)}")
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                causal_states = batch['causal_state']
                
                result = model(input_ids, attention_mask, causal_states)
                loss = result['contrastive_loss']
                
                if not torch.isnan(loss):
                    val_loss += loss.item()
                    val_batches += 1
        
        avg_val_loss = val_loss / val_batches if val_batches > 0 else 0
        
        # Evaluate embeddings every 5 epochs
        separation_score = 0
        intra_sim = 0 
        inter_sim = 0
        
        if epoch % 5 == 0 or epoch == num_epochs - 1:
            print(f"📊 Evaluating embeddings...")
            metrics = evaluate_embeddings_fsm(model, val_dataloader, device, num_samples=100)
            
            separation_score = metrics.get('separation_score', 0)
            intra_sim = metrics.get('intra_class_similarity', 0)
            inter_sim = metrics.get('inter_class_similarity', 0)
            state_dist = metrics.get('state_distribution', {})
            
            # Save best model
            if separation_score > best_separation:
                best_separation = separation_score
                patience_counter = 0
                
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'vocab_info': vocab_info,
                    'metrics': metrics,
                    'epoch': epoch,
                    'config': {
                        'd_model': 64,
                        'embedding_dim': 32,
                        'num_layers': 2,
                        'learning_rate': learning_rate
                    }
                }, output_dir / 'best_model.pt')
                
                print(f"    💾 New best! Sep={separation_score:.4f}")
            else:
                patience_counter += 1
            
            print(f"    📊 Metrics: Sep={separation_score:.4f}, Intra={intra_sim:.4f}, Inter={inter_sim:.4f}")
            print(f"    🎯 States found: {state_dist}")
        
        # Record history
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['separation_score'].append(separation_score)
        history['intra_class_similarity'].append(intra_sim)
        history['inter_class_similarity'].append(inter_sim)
        
        # Learning rate scheduling
        scheduler.step(separation_score)
        
        # Print progress
        epoch_time = time.time() - start_time
        print(f"Epoch {epoch+1:2d}/{num_epochs} | "
              f"Loss: {avg_train_loss:.4f}/{avg_val_loss:.4f} | "
              f"Sep: {separation_score:.4f} | "
              f"Patience: {patience_counter}/{max_patience} | "
              f"{epoch_time:.1f}s")
        
        # Early stopping
        if patience_counter >= max_patience:
            print(f"Early stopping after {epoch+1} epochs")
            break
    
    print(f"\n✅ Training completed!")
    print(f"Best separation score: {best_separation:.4f}")
    
    # Final analysis
    print(f"\n🧪 Final analysis...")
    
    if (output_dir / 'best_model.pt').exists():
        checkpoint = torch.load(output_dir / 'best_model.pt', map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Final metrics
        final_metrics = evaluate_embeddings_fsm(model, val_dataloader, device, num_samples=200)
        print(f"Final metrics:")
        print(f"  Separation score: {final_metrics['separation_score']:.4f}")
        print(f"  Intra-class similarity: {final_metrics['intra_class_similarity']:.4f}")
        print(f"  Inter-class similarity: {final_metrics['inter_class_similarity']:.4f}")
        print(f"  States discovered: {final_metrics['state_distribution']}")
        
        # Test specific sequences
        model.eval()
        epsilon_machine = EpsilonMachine()
        
        test_sequences = ['0', '1', '00', '01', '10', '11', '000', '001', '010', '011', '100', '101', '110', '111']
        
        print(f"\n🔍 Testing specific sequences:")
        with torch.no_grad():
            for seq in test_sequences:
                # Get true causal state
                true_state = epsilon_machine.get_causal_state(list(seq))
                
                # Convert to model input
                input_ids = [int(c) for c in seq]
                attention_mask = [1] * len(input_ids)
                
                # Pad to max_length
                padded_length = 20
                attention_mask = attention_mask + [0] * (padded_length - len(attention_mask))
                input_ids = input_ids + [2] * (padded_length - len(input_ids))
                
                input_tensor = torch.tensor([input_ids], device=device)
                mask_tensor = torch.tensor([attention_mask], device=device)
                
                result = model(input_tensor, mask_tensor)
                embedding = result['embeddings'][0]
                
                if not torch.isnan(embedding).any():
                    norm = embedding.norm().item()
                    print(f"  '{seq}' (state {true_state}) → norm: {norm:.3f}")
    
    # Save complete results
    results = {
        'history': history,
        'best_separation_score': best_separation,
        'final_metrics': final_metrics if 'final_metrics' in locals() else {},
        'config': {
            'batch_size': batch_size,
            'learning_rate': learning_rate,
            'num_epochs': num_epochs,
            'model_config': {
                'd_model': 64,
                'embedding_dim': 32,
                'num_layers': 2
            }
        }
    }
    
    with open(output_dir / 'training_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n💾 Results saved to {output_dir}/")
    
    return results


if __name__ == "__main__":
    train_fsm_contrastive()