"""
Feature extraction from nanoGPT transformer layers.

Extracts activations from all transformer blocks and the LM head input.
Supports extracting from residual stream pre/post MLP and pre/post attention.
"""

import logging
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Any, Callable
from contextlib import contextmanager
import numpy as np


class FeatureExtractor:
    """
    Hooks into nanoGPT to extract intermediate activations.

    Can extract from:
    - Final token representations at each transformer block
    - Pre/post attention activations
    - Pre/post MLP activations
    - LM head input (pre-logits)
    """

    def __init__(self, model: nn.Module):
        """
        Initialize with a nanoGPT model.

        Args:
            model: Trained GPT model
        """
        self.logger = logging.getLogger(self.__class__.__name__)

        self.model = model
        self.config = model.config
        self.hooks = []
        self.captured_activations = {}

        # Identify hook points
        self.hook_points = self._get_hook_points()

    def _get_hook_points(self) -> Dict[str, str]:
        """Get mapping of layer names to descriptive names."""
        hook_points = {}

        # Transformer blocks
        for i in range(self.config.n_layer):
            hook_points[f'transformer.h.{i}'] = f'block_{i}'
            # Optional: pre/post attention and MLP
            # hook_points[f'transformer.h.{i}.ln_1'] = f'block_{i}_pre_attn'
            # hook_points[f'transformer.h.{i}.attn'] = f'block_{i}_post_attn'
            # hook_points[f'transformer.h.{i}.ln_2'] = f'block_{i}_pre_mlp'
            # hook_points[f'transformer.h.{i}.mlp'] = f'block_{i}_post_mlp'

        # Final layer norm (input to LM head)
        hook_points['transformer.ln_f'] = 'lm_head_input'

        return hook_points

    def _install_hooks(self):
        """Install forward hooks to capture activations."""
        self.captured_activations = {}

        def hook_fn(name: str):
            def hook(module, input, output):
                # For transformer blocks, capture the final output (residual + attention + MLP)
                # For ln_f, capture the normalized output (input to LM head)
                if isinstance(output, torch.Tensor):
                    # Take final token representation (last position for each sequence)
                    # Shape: (batch, seq_len, hidden) -> (batch, hidden)
                    final_tokens = output[:, -1, :]  # [batch, hidden]
                    self.captured_activations[name] = final_tokens.detach().cpu().numpy()
                else:
                    # Handle tuple outputs if any
                    self.captured_activations[name] = output

            return hook

        # Install hooks
        for module_path, name in self.hook_points.items():
            module = self._get_module_by_path(module_path)
            if module is not None:
                hook_handle = module.register_forward_hook(hook_fn(name))
                self.hooks.append(hook_handle)

    def _get_module_by_path(self, path: str) -> Optional[nn.Module]:
        """Get module by dot-separated path."""
        parts = path.split('.')
        module = self.model

        for part in parts:
            if hasattr(module, part):
                module = getattr(module, part)
            else:
                return None

        return module

    def _remove_hooks(self):
        """Remove all installed hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []

    @contextmanager
    def capture_activations(self):
        """Context manager for capturing activations during forward pass."""
        self._install_hooks()
        try:
            yield self.captured_activations
        finally:
            self._remove_hooks()

    def extract_batch(self, batch_input: torch.Tensor,
                     return_dict: bool = True) -> Dict[str, np.ndarray]:
        """
        Extract activations for a batch of inputs.

        Args:
            batch_input: Input tensor of shape (batch, seq_len)
            return_dict: If True, return dict of layer_name -> activations
                        If False, return list of activations in layer order

        Returns:
            Dict or list of numpy arrays with shape (batch, hidden_dim)
        """
        self.model.eval()

        with torch.no_grad(), self.capture_activations() as activations:
            # Forward pass - model returns (logits, loss) tuple
            _ = self.model(batch_input)[0]  # Just get logits

        # Format output
        if return_dict:
            self.logger.debug(
                "Captured activations for %d layers (batch=%d, seq_len=%d)",
                len(activations), batch_input.shape[0], batch_input.shape[1]
            )
            return dict(activations)
        else:
            # Return in layer order: block_0, block_1, ..., lm_head_input
            ordered_names = [f'block_{i}' for i in range(self.config.n_layer)] + ['lm_head_input']
            return [activations[name] for name in ordered_names]

    def extract_dataset(self, dataloader: Any,
                       batch_size: int = 32,
                       max_samples: Optional[int] = None) -> Dict[str, np.ndarray]:
        """
        Extract activations for an entire dataset.

        Args:
            dataloader: DataLoader yielding batches of input tensors
            batch_size: Batch size for extraction
            max_samples: Maximum samples to extract (None for all)

        Returns:
            Dict of layer_name -> (N, hidden_dim) activations
        """
        all_activations = {}
        total_samples = 0

        for batch in dataloader:
            if isinstance(batch, dict):
                batch_input = batch['input_ids']  # Standard HF format
            else:
                batch_input = batch  # Direct tensor

            batch_acts = self.extract_batch(batch_input)

            # Accumulate activations
            for layer_name, acts in batch_acts.items():
                if layer_name not in all_activations:
                    all_activations[layer_name] = []
                all_activations[layer_name].append(acts)

            total_samples += len(batch_input)
            if max_samples and total_samples >= max_samples:
                break

        # Concatenate all batches
        final_activations = {}
        for layer_name, act_list in all_activations.items():
            final_activations[layer_name] = np.concatenate(act_list, axis=0)

            if max_samples:
                final_activations[layer_name] = final_activations[layer_name][:max_samples]

        return final_activations

    def get_layer_names(self) -> List[str]:
        """Get list of layer names in order."""
        return [f'block_{i}' for i in range(self.config.n_layer)] + ['lm_head_input']

    def get_hidden_dim(self) -> int:
        """Get hidden dimension of the model."""
        return self.config.n_embd
