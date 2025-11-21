import torch
import sys
from pathlib import Path
import numpy as np

# Add repo root to path
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "nanoGPT"))
sys.path.insert(0, str(REPO_ROOT / "cssr_discovery"))

from nanoGPT.model import GPT, GPTConfig
from cssr_discovery.two_stage_oracle_cssr import precompute_predictions

def test_cssr_integration():
    print("Setting up model...")
    config = GPTConfig(
        block_size=64,
        vocab_size=2, # Use 2 to match binary history assumption and avoid explosion
        n_layer=2,
        n_head=2,
        n_embd=32,
        dropout=0.0,
        bias=True
    )
    model = GPT(config)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    model.to(device)
    model.eval()

    # Create dummy histories
    # Histories are np.ndarray of shape (L,)
    L = 10
    num_histories = 5
    histories = [np.random.randint(0, 2, size=(L,)) for _ in range(num_histories)]
    
    metrics_k = [1, 3, 5]
    
    print("Running precompute_predictions with KV cache...")
    try:
        preds = precompute_predictions(
            histories=histories,
            metrics_k=metrics_k,
            model=model,
            platt_params=None,
            pred_batch_size=2, # Small batch size to test chunking
            branch_topk=2 # Limit branching to avoid explosion
        )
        print("Success! Predictions computed.")
    except Exception as e:
        print(f"FAILED with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # Basic validation
    for k in metrics_k:
        if k not in preds:
            print(f"Missing k={k} in predictions")
            sys.exit(1)
        print(f"k={k} predictions present for {len(preds[k])} histories")
        
        # Check shape of a prediction
        # Should be (2^k,)
        first_key = next(iter(preds[k]))
        arr = preds[k][first_key]
        if arr.shape != (1 << k,):
            print(f"Bad shape for k={k}: {arr.shape}, expected {(1 << k,)}")
            sys.exit(1)

    print("Integration test passed.")

if __name__ == "__main__":
    test_cssr_integration()
