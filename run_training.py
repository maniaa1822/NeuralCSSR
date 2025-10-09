"""Convenience entry point for supervised training.

Example commands:

```bash
uv run --with numpy --with torch run_training.py \
  --epochs 3 --batch_size 6 --lr 3e-4 \
  --checkpoint results/eps_decoder.pt \
  --log_dir results/tb_logs

uv run --with numpy --with torch run_training.py \

uv run --with numpy python - <<'PY'
import numpy as np
from pathlib import Path
from ood_machine_generator import dataset_iter

num_machines = 200
seqs_per_machine = 500
length = 1000
pad_K = 12
rng = 1337
  --epochs 1 --batch_size 4 --device cpu
iterator = dataset_iter(
    n_machines=num_machines,
    seqs_per_machine=seqs_per_machine,
    lengths=[length],
    pad_K=pad_K,
    rng=rng,
    include_states=False,
)

samples = []
for sample in iterator:
    samples.append(sample)
    if len(samples) >= num_machines * seqs_per_machine:
        break

seqs = np.stack([s.seq.astype(np.uint8) for s in samples])
lengths = np.array([s.length for s in samples], dtype=np.int32)
E = np.stack([s.padded_emissions.astype(np.float32) for s in samples])
T = np.stack([s.padded_transitions.astype(np.int16) for s in samples])
mask = np.stack([s.state_mask.astype(bool) for s in samples])

output_path = Path("results/ood_supervised_dataset_200machines_100M.npz")
np.savez_compressed(
    output_path,
    tokens=seqs,
    lengths=lengths,
    emissions=E,
    transitions=T,
    mask=mask,
    machine_ids=np.array([s.machine_id for s in samples]),
    splits=np.array([s.split for s in samples]),
    anchor_views=np.array([s.anchor_view for s in samples], dtype=np.int16),
    pair_keys=np.array([s.pair_key for s in samples]),
)
print(f"Saved {seqs.size} tokens to {output_path}")
PY
uv run --with numpy --with torch run_training.py \
  --epochs 5 --batch_size 16 --lr 2e-4 \
  --checkpoint results/eps_decoder_large.pt \
  --log_dir results/tb_large \
  --val_fraction 0.1

uv run --with numpy --with torch run_training.py \
  --dataset results/ood_supervised_dataset_200machines_10M.npz \
  --train_split train --val_split val \
  --epochs 50 --batch_size 32 --lr 2e-4 \
  --checkpoint results/eps_decoder_10M.pt \
  --log_dir results/tb_10M
```
"""

from ood_supervised.run_training import main


if __name__ == "__main__":
    main()
