#!/usr/bin/env python3
"""
Test the complete FSM transformer pipeline.
"""

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import json
from pathlib import Path

from fsm_transformer import (
    EpsilonMachine, 
    EpsilonMachineDataGenerator, 
    AutoregressiveTransformer,
    load_dataset
)


def test_epsilon_machine():
    """Test epsilon machine implementation."""
    print("Testing epsilon machine...")
    
    machine = EpsilonMachine(seed=42)
    
    # Test sequence generation
    sequence, states = machine.generate_sequence(20)
    print(f"Generated sequence: {''.join(sequence)}")
    print(f"State trajectory: {states}")
    
    # Test causal state mapping
    test_histories = [[], ['0'], ['1'], ['0', '0'], ['1', '1'], ['0', '1']]
    for history in test_histories:
        state = machine.get_causal_state(history)
        print(f"History {''.join(history)} -> State {state}")
    
    # Test stationary distribution
    stationary = machine.compute_stationary_distribution()
    print(f"Stationary distribution: {stationary}")
    
    print("✓ Epsilon machine tests passed\n")


def test_data_generation():
    """Test data generation."""
    print("Testing data generation...")
    
    generator = EpsilonMachineDataGenerator(seed=42)
    
    # Generate small dataset
    dataset, metadata = generator.generate_training_data(
        num_sequences=100,
        min_length=10,
        max_length=20,
        output_dir="data/test_fsm"
    )
    
    print(f"Dataset size: {len(dataset)}")
    print(f"Vocabulary: {dataset.token_to_id}")
    
    # Test dataloader
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)
    batch = next(iter(dataloader))
    
    print(f"Batch keys: {batch.keys()}")
    print(f"Input shape: {batch['input_ids'].shape}")
    print(f"Target shape: {batch['target_id'].shape}")
    
    print("✓ Data generation tests passed\n")


def test_transformer():
    """Test transformer implementation."""
    print("Testing transformer...")
    
    model = AutoregressiveTransformer(
        vocab_size=3,
        embed_dim=32,  # Smaller for testing
        num_heads=2,
        num_layers=2,
        ff_dim=64,
        max_seq_len=128
    )
    
    # Test forward pass
    batch_size, seq_len = 4, 10
    input_ids = torch.randint(0, 3, (batch_size, seq_len))
    attention_mask = torch.ones_like(input_ids)
    
    outputs = model(input_ids, attention_mask)
    print(f"Output logits shape: {outputs['logits'].shape}")
    print(f"Hidden state shape: {outputs['last_hidden_state'].shape}")
    
    # Test loss computation
    target_ids = torch.randint(0, 2, (batch_size,))  # Only tokens 0 and 1 for targets
    loss = model.compute_loss(input_ids, target_ids, attention_mask)
    print(f"Loss: {loss.item():.4f}")
    
    # Test probability computation
    probs = model.get_next_token_probabilities(input_ids, attention_mask)
    print(f"Next token probabilities shape: {probs.shape}")
    print(f"Sample probabilities: {probs[0]}")
    
    print("✓ Transformer tests passed\n")


def test_training_loop():
    """Test simple training loop."""
    print("Testing training loop...")
    
    # Load test dataset
    dataset, metadata = load_dataset("data/test_fsm")
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
    
    # Initialize model
    model = AutoregressiveTransformer(
        vocab_size=dataset.vocab_size,
        embed_dim=32,
        num_heads=2,
        num_layers=2,
        ff_dim=64
    )
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    # Train for a few steps
    model.train()
    total_loss = 0
    num_batches = 5
    
    for i, batch in enumerate(dataloader):
        if i >= num_batches:
            break
            
        optimizer.zero_grad()
        
        input_ids = batch['input_ids']
        target_ids = batch['target_id']
        attention_mask = batch['attention_mask']
        
        loss = model.compute_loss(input_ids, target_ids, attention_mask)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        if i % 2 == 0:
            print(f"Step {i}: Loss = {loss.item():.4f}")
    
    avg_loss = total_loss / num_batches
    print(f"Average loss over {num_batches} batches: {avg_loss:.4f}")
    
    print("✓ Training loop tests passed\n")


def test_analysis_setup():
    """Test analysis components."""
    print("Testing analysis setup...")
    
    # Load test data
    with open("data/test_fsm/raw_data.json", 'r') as f:
        raw_data = json.load(f)
    
    sequences = raw_data['sequences'][:10]  # Small sample
    states = raw_data['state_trajectories'][:10]
    
    print(f"Loaded {len(sequences)} test sequences")
    print(f"Sample sequence: {''.join(sequences[0])}")
    print(f"Sample states: {states[0]}")
    
    # Test representation extraction (without full analysis)
    model = AutoregressiveTransformer(vocab_size=3, embed_dim=32, num_heads=2, num_layers=2)
    
    # Convert first sequence to input
    sequence = sequences[0][:10]  # Truncate for test
    input_ids = torch.tensor([[int(token) for token in sequence]], dtype=torch.long)
    attention_mask = torch.ones_like(input_ids)
    
    representations = model.extract_representations(input_ids, attention_mask)
    print(f"Extracted representations shape: {representations.shape}")
    
    print("✓ Analysis setup tests passed\n")


def main():
    """Run all tests."""
    print("=== FSM Transformer Pipeline Tests ===\n")
    
    test_epsilon_machine()
    test_data_generation()
    test_transformer()
    test_training_loop()
    test_analysis_setup()
    
    print("=== All Tests Passed! ===")
    print("\nPipeline is ready for:")
    print("1. Large-scale data generation (50k sequences)")
    print("2. Full transformer training")
    print("3. Causal state analysis and clustering")
    print("4. Comparison with true epsilon-machine structure")


if __name__ == '__main__':
    main()