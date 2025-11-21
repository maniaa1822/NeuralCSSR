import torch
import sys
from pathlib import Path
import numpy as np

# Add repo root to path
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "nanoGPT"))

from nanoGPT.model import GPT, GPTConfig

def test_kv_cache():
    # Create a small model
    config = GPTConfig(
        block_size=64,
        vocab_size=100,
        n_layer=2,
        n_head=2,
        n_embd=32,
        dropout=0.0,
        bias=True
    )
    model = GPT(config)
    model.eval()

    # Create a random sequence
    seq_len = 10
    input_ids = torch.randint(0, 100, (1, seq_len))

    # 1. Standard forward pass (no cache)
    # By default model() returns only the last logit if targets is None.
    # We need to pass targets (dummy) or modify model to return all?
    # Or just loop over positions for reference too?
    # Actually, let's just use targets=input_ids to get full sequence logits (shifted? no, standard forward returns logits)
    # Wait, if targets is not None, it returns loss.
    # Let's modify the call to force full sequence return or just loop.
    
    logits_ref_list = []
    with torch.no_grad():
        # To get full sequence logits from this specific model implementation without targets,
        # we need to pass last_position_indices for every position? No that's inefficient.
        # The model implementation gathers:
        # gathered = x[torch.arange(b, device=device), token_positions]
        # If we want all, we should probably just use the fact that if we pass targets, we get logits for all positions?
        # "logits = self.lm_head(x)" happens if targets is not None.
        # So let's pass dummy targets.
        
        dummy_targets = input_ids.clone()
        logits_ref, _, _ = model(input_ids, targets=dummy_targets)
        # This returns logits of shape (B, T, V) and loss.
        
    # logits_ref is (1, 10, 100) now.

    # 2. Incremental forward pass (with cache)
    past_key_values = None
    logits_cached = []
    
    with torch.no_grad():
        # Feed tokens one by one
        for i in range(seq_len):
            token = input_ids[:, i:i+1]
            
            # If first token, pass full context so far? 
            # No, KV cache usage pattern:
            # Step 0: pass token 0. Get KV0.
            # Step 1: pass token 1, past=KV0. Get KV1.
            
            # Wait, standard usage is:
            # Prefill: pass tokens [0...N]. Get KV_N.
            # Decode: pass token N+1, past=KV_N.
            
            # Let's test pure incremental generation from scratch
            if past_key_values is None:
                # First step
                out_logits, _, past_key_values = model(token, past_key_values=None)
            else:
                out_logits, _, past_key_values = model(token, past_key_values=past_key_values)
            
            logits_cached.append(out_logits)

    # Compare logits
    # logits_ref is (1, seq_len, vocab_size)
    # logits_cached is list of (1, 1, vocab_size)
    
    logits_cached_stack = torch.cat(logits_cached, dim=1)
    
    print(f"Ref shape: {logits_ref.shape}")
    print(f"Cached shape: {logits_cached_stack.shape}")
    
    diff = (logits_ref - logits_cached_stack).abs().max().item()
    print(f"Max difference: {diff}")
    
    if diff < 1e-5:
        print("SUCCESS: KV cache matches standard forward pass.")
    else:
        print("FAILURE: Mismatch detected.")
        sys.exit(1)

    # 3. Test mixed usage (Prefill + Decode)
    print("\nTesting Prefill + Decode...")
    prefill_len = 5
    with torch.no_grad():
        # Prefill
        prefill_ids = input_ids[:, :prefill_len]
        logits_pre, _, past_key_values = model(prefill_ids, past_key_values=None)
        
        # Decode remaining
        logits_decode = []
        for i in range(prefill_len, seq_len):
            token = input_ids[:, i:i+1]
            out_logits, _, past_key_values = model(token, past_key_values=past_key_values)
            logits_decode.append(out_logits)
            
    logits_decode_stack = torch.cat(logits_decode, dim=1)
    # Compare with ref logits for the decoded part
    logits_ref_decode = logits_ref[:, prefill_len:, :]
    # Note: logits_pre corresponds to positions 0..4. logits_decode corresponds to positions 5..9?
    # Wait, model(x) returns logits for the input positions.
    # So logits_decode[0] is the logit for input_ids[prefill_len].
    # This matches logits_ref[:, prefill_len:prefill_len+1, :]
    
    diff_decode = (logits_ref_decode - logits_decode_stack).abs().max().item()
    print(f"Max difference (decode): {diff_decode}")
    
    if diff_decode < 1e-5:
        print("SUCCESS: Prefill + Decode matches reference.")
    else:
        print("FAILURE: Prefill + Decode mismatch.")
        sys.exit(1)

if __name__ == "__main__":
    test_kv_cache()
