from __future__ import annotations

import json
import numpy as np
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import torch
from torch.utils.data import Dataset

from ood_machine_generator import dataset_iter, sample_to_payload


@dataclass
class MachineDatasetConfig:
    n_machines: int
    seqs_per_machine: int
    lengths: Sequence[int]
    pad_K: int
    rng: int
    include_states: bool = False
    machine_kwargs: Optional[Dict] = None
    split_fraction: Optional[float] = None


class MachineDataset(Dataset):
    def __init__(self, samples: List[Dict]):
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict:
        return self.samples[idx]


def generate_dataset(config: MachineDatasetConfig) -> MachineDataset:
    iterator = dataset_iter(
        n_machines=config.n_machines,
        seqs_per_machine=config.seqs_per_machine,
        lengths=config.lengths,
        pad_K=config.pad_K,
        rng=config.rng,
        include_states=config.include_states,
        machine_kwargs=config.machine_kwargs,
    )

    samples: List[Dict] = []
    limit = config.n_machines * config.seqs_per_machine
    for sample in iterator:
        payload = sample_to_payload(sample)
        samples.append(payload)
        if len(samples) >= limit:
            break

    return MachineDataset(samples)


def split_dataset(dataset: MachineDataset, fraction: float) -> tuple[MachineDataset, MachineDataset]:
    if not 0.0 < fraction < 1.0:
        raise ValueError("fraction must be in (0, 1)")
    split_idx = int(len(dataset) * fraction)
    samples = dataset.samples
    return MachineDataset(samples[:split_idx]), MachineDataset(samples[split_idx:])


def subset_by_split(dataset: MachineDataset, split: str) -> MachineDataset:
    filtered = [sample for sample in dataset.samples if sample.get("split") == split]
    return MachineDataset(filtered)


def load_preview_dataset(path: Path) -> MachineDataset:
    raw = json.loads(path.read_text())
    return MachineDataset(raw)


def collate_sequences(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    max_len = max(item["length"] for item in batch)
    pad_K = batch[0]["padded"]["K_max"]

    tokens = torch.zeros(len(batch), max_len, dtype=torch.long)
    attention_mask = torch.zeros(len(batch), max_len, dtype=torch.bool)
    emissions = torch.zeros(len(batch), pad_K)
    transitions = torch.zeros(len(batch), pad_K, 2, dtype=torch.long)
    masks = torch.zeros(len(batch), pad_K)

    machine_ids: List[Optional[str]] = []
    splits: List[Optional[str]] = []

    for idx, item in enumerate(batch):
        seq = item["seq"]
        length = item["length"]
        tokens[idx, :length] = torch.tensor(seq, dtype=torch.long)
        attention_mask[idx, :length] = True
        emissions[idx] = torch.tensor(item["padded"]["E"], dtype=torch.float32)
        transitions[idx] = torch.tensor(item["padded"]["T"], dtype=torch.long)
        masks[idx] = torch.tensor(item["padded"]["state_mask"], dtype=torch.float32)
        machine_ids.append(item.get("machine_id"))
        splits.append(item.get("split"))

    batch_dict: Dict[str, torch.Tensor | List[Optional[str]]] = {
        "tokens": tokens,
        "attention_mask": attention_mask,
        "emissions": emissions,
        "transitions": transitions,
        "mask": masks,
    }


    if any(mid is not None for mid in machine_ids):
        batch_dict["machine_id"] = [mid if mid is None else str(mid) for mid in machine_ids]
    if any(s is not None for s in splits):
        batch_dict["split"] = [s if s is None else str(s) for s in splits]

    return batch_dict


def load_npz_dataset(path: Path) -> MachineDataset:
    data = np.load(path, allow_pickle=True)
    tokens = data["tokens"]
    lengths = data["lengths"]
    emissions = data["emissions"]
    transitions = data["transitions"]
    mask = data["mask"]
    machine_ids = data.get("machine_ids")
    splits = data.get("splits")
    pad_K = emissions.shape[1]

    samples: List[Dict] = []
    num_samples = tokens.shape[0]
    for idx in range(num_samples):
        sample = {
            "machine_id": str(machine_ids[idx]) if machine_ids is not None else None,
            "split": str(splits[idx]) if splits is not None else None,
            "seq": tokens[idx].astype(int).tolist(),
            "length": int(lengths[idx]),
            "padded": {
                "K_max": pad_K,
                "E": emissions[idx].astype(float).tolist(),
                "T": transitions[idx].astype(int).tolist(),
                "state_mask": mask[idx].astype(bool).tolist(),
            },
        }
        samples.append(sample)

    return MachineDataset(samples)