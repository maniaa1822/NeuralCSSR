"""
Test model performance on actual training data subsequences.
This should match the validation loss if our hypothesis is correct.
"""

import sys
from pathlib import Path
import numpy as np
import torch
from contextlib import nullcontext

sys.path.insert(0, str(Path(__file__).parent.parent))

from nanoGPT.model import GPT, GPTConfig

# Load model
ckpt_path = 'nanoGPT/out-seven-state-human-char-icl/ckpt.pt'
device = 'cuda'

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

# Load training data
data = np.memmap('nanoGPT/data/seven_state_human/train.bin', dtype=np.uint16, mode='r')
print(f"Training data size: {len(data)} tokens")

# Test on random subsequences from training data
n_samples = 100
block_size = 32
cross_entropies = []
accuracies = []

dtype = torch.float16
ctx = torch.amp.autocast(device_type='cuda', dtype=dtype)

for _ in range(n_samples):
    # Random position from training data
    idx = np.random.randint(0, len(data) - block_size - 1)

    # Get context and true next token
    context = torch.from_numpy(data[idx:idx+block_size].astype(np.int64))[None, ...].to(device)
    true_next = int(data[idx + block_size])

    # Get model prediction
    with torch.no_grad():
        with ctx:
            logits, _ = model(context)
            probs = torch.softmax(logits[:, -1, :], dim=-1).cpu().numpy()[0]

    # Metrics
    true_prob = probs[true_next]
    cross_entropies.append(-np.log2(true_prob + 1e-10))
    accuracies.append(1.0 if np.argmax(probs) == true_next else 0.0)

print(f"\nTest on training data subsequences (block_size={block_size}):")
print(f"  Mean cross-entropy: {np.mean(cross_entropies):.4f} bits")
print(f"  Top-1 accuracy: {np.mean(accuracies):.2%}")
print(f"\nCompare to:")
print(f"  Training val loss: 0.771 bits")
print(f"  ICL eval baseline: 0.973 bits")
