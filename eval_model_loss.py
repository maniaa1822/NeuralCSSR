import torch
import numpy as np
import sys
from pathlib import Path
import math

# Add nanoGPT to path
sys.path.insert(0, str(Path(__file__).parent / 'nanoGPT'))
from model import GPT, GPTConfig

def load_model(ckpt_path, device='cuda'):
    checkpoint = torch.load(ckpt_path, map_location=device)
    gptconf = GPTConfig(**checkpoint['model_args'])
    model = GPT(gptconf)
    
    state_dict = checkpoint['model']
    unwanted_prefix = '_orig_mod.'
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
            
    model.load_state_dict(state_dict)
    model.eval()
    model.to(device)
    return model, gptconf.block_size

def compute_nll(model, data_path, block_size, device='cuda'):
    # Load data
    with open(data_path, 'r') as f:
        raw = f.read()
    data = ''.join(ch for ch in raw if ch in ('0', '1'))
    data_tokens = torch.tensor([int(c) for c in data], dtype=torch.long)
    
    n = len(data_tokens)
    total_loss = 0.0
    steps = 0
    
    batch_size = 512
    
    print(f"Evaluating on {n} tokens...")
    
    # We need to evaluate P(x_t | x_{<t})
    # We can process in batches.
    # For a sequence of length N, we can feed it into the model.
    # However, the model has a block_size limit.
    # We can use a sliding window or just batch chunks if we don't care about state reset (but we do for RNNs/Transformers).
    # Actually, for a Transformer, we can just feed chunks of block_size.
    # But we want to evaluate *every* token position given its context.
    
    # Simple approach: Iterate through the data with a context window.
    # Efficient approach: Use batches of sliding windows? No, that's O(N*block_size).
    # Standard approach for LM eval: Chunk the data into segments of block_size.
    # Loss at position i depends on 0..i-1.
    # If we chunk [0:block_size], we get losses for 1..block_size.
    # Then [block_size:2*block_size]... but we lose context from the first chunk!
    # For a Transformer, context is limited to block_size anyway.
    # So chunking is fine, as long as we overlap?
    # No, standard LM eval usually just chunks and accepts the context break at boundaries,
    # OR uses a sliding window with stride < block_size.
    
    # Let's use a stride of block_size for simplicity and speed, consistent with training.
    # This means predictions at the start of a chunk have limited context.
    # But this is exactly how the model was trained (likely).
    
    # Wait, for CSSR we used L_max=6. The model has block_size=64 or 128 likely.
    # The Epsilon Machine loss was computed using L=6 context.
    # To be fair, we should perhaps restrict the model to L=6 context too?
    # The user asked: "recover the same loss as the model during training".
    # During training, the model uses up to block_size context.
    # So we should measure the model's *best* performance (full context).
    
    data_tensor = data_tokens.to(device)
    
    with torch.no_grad():
        for i in range(0, n - 1, block_size):
            # Input: data[i : i+block_size]
            # Target: data[i+1 : i+block_size+1]
            
            x = data_tensor[i : i + block_size].unsqueeze(0)
            y = data_tensor[i+1 : i + block_size + 1].unsqueeze(0)
            
            if y.size(1) < x.size(1):
                x = x[:, :y.size(1)]
            
            if x.size(1) < 1:
                break
                
            # Forward pass
            # model(x, y) returns (logits, loss, new_key_values)
            _, loss, _ = model(x, y)
            
            # We need total NLL, so multiply by number of tokens
            batch_tokens = x.size(1)
            total_loss += loss.item() * batch_tokens
            steps += batch_tokens
            
    avg_nll = total_loss / steps
    avg_bits = avg_nll / math.log(2)
    
    print(f"Total tokens evaluated: {steps}")
    print(f"Average NLL: {avg_nll:.4f} nats")
    print(f"Average Bits: {avg_bits:.4f} bits")
    
    return avg_nll

if __name__ == '__main__':
    model_path = 'nanoGPT/out-gm-seven-union-char/ckpt.pt'
    data_path = 'experiments/datasets/gm_seven_union/combined.dat'
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    model, block_size = load_model(model_path, device)
    print(f"Model block size: {block_size}")
    
    compute_nll(model, data_path, block_size, device)
