"""
Main entanglement analysis pipeline.

Orchestrates data extraction, feature extraction, probing, and metric computation.
"""

import logging
import numpy as np
import torch
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
import json
import pickle
from datetime import datetime

from .data import DataExtractor, PrimitiveFactors
from .features import FeatureExtractor
from .probes import LinearProbeSuite, MLPProbeSuite, CompositionEvaluator, ProbeEvaluator
from .metrics import EntanglementMetrics, FragmentationAnalyzer, TrajectoryAnalyzer, FERCalculator
from .utils.theoretical_limits import compute_next_token_bayes_accuracy


class EntanglementAnalysisPipeline:
    """
    Complete pipeline for quantifying entanglement in nanoGPT representations.

    Follows the methodology from the specification:
    1. Extract data and features
    2. Train linear and non-linear probes
    3. Evaluate compositionality
    4. Compute entanglement metrics
    5. Generate FER scores
    """

    def __init__(self, machine, model_path: str, output_dir: str = "./results"):
        """
        Initialize the analysis pipeline.

        Args:
            machine: ε-machine specification
            model_path: Path to trained nanoGPT model
            output_dir: Directory to save results
        """
        self.logger = logging.getLogger(self.__class__.__name__)

        self.machine = machine
        self.model_path = model_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        # Theoretical limits
        self.next_token_bayes_accuracy = compute_next_token_bayes_accuracy(machine)
        self.logger.info(
            "Machine '%s' Bayes next-token accuracy=%.3f",
            self.machine.name,
            self.next_token_bayes_accuracy,
        )

        # Initialize components
        self.data_extractor = DataExtractor(machine)
        self.primitive_info = PrimitiveFactors.get_primitive_info()

        # Load model
        self.model = self._load_model()
        self.feature_extractor = FeatureExtractor(self.model)

        # Initialize analyzers
        self.linear_probes = LinearProbeSuite()
        self.mlp_probes = MLPProbeSuite()
        self.composition_evaluator = CompositionEvaluator(self.primitive_info)
        self.probe_evaluator = ProbeEvaluator()
        self.entanglement_metrics = EntanglementMetrics()
        self.fragmentation_analyzer = FragmentationAnalyzer()
        self.trajectory_analyzer = TrajectoryAnalyzer()
        self.fer_calculator = FERCalculator()

    def _load_model(self):
        """Load trained nanoGPT model."""
        # Import here to avoid circular imports
        from nanoGPT.model import GPT, GPTConfig

        checkpoint = torch.load(self.model_path, map_location='cpu')
        model_args = checkpoint.get('model_args')

        if model_args is None:
            cfg = checkpoint.get('config', {})
            required_keys = ['n_layer', 'n_head', 'n_embd', 'block_size']
            if not all(k in cfg for k in required_keys):
                raise KeyError("Checkpoint missing model_args and insufficient config to rebuild model")

            vocab_size = cfg.get('vocab_size')
            dataset_name = cfg.get('dataset')
            if vocab_size is None and dataset_name:
                meta_path = Path('nanoGPT') / 'data' / dataset_name / 'meta.pkl'
                if meta_path.exists():
                    with open(meta_path, 'rb') as f:
                        meta = pickle.load(f)
                        vocab_size = meta.get('vocab_size')
            if vocab_size is None:
                raise KeyError("Unable to determine vocab_size for multitask checkpoint")

            model_args = {
                'n_layer': cfg['n_layer'],
                'n_head': cfg['n_head'],
                'n_embd': cfg['n_embd'],
                'block_size': cfg['block_size'],
                'bias': cfg.get('bias', False),
                'dropout': cfg.get('dropout', 0.0),
                'vocab_size': vocab_size,
            }

        config = GPTConfig(**model_args)
        model = GPT(config)

        state_dict = checkpoint['model']
        # Unwrap compiled state dicts
        if any(key.startswith('_orig_mod.') for key in state_dict.keys()):
            state_dict = {key.replace('_orig_mod.', ''): value
                          for key, value in state_dict.items()}

        # Handle multitask checkpoints that namespace base weights under 'gpt.'
        if any(key.startswith('gpt.') for key in state_dict.keys()):
            remapped = {}
            for key, value in state_dict.items():
                if key.startswith('gpt.'):
                    remapped[key.replace('gpt.', '', 1)] = value
                elif key.startswith('transformer.') or key.startswith('lm_head.'):
                    # Already base model keys; keep as-is
                    remapped[key] = value
                else:
                    # Ignore auxiliary heads (e.g., epsilon_heads.*)
                    continue
            state_dict = remapped

        model.load_state_dict(state_dict, strict=False)
        model.eval()

        return model

    def run_full_analysis(self, sequences: List[np.ndarray],
                         n_samples: int = 10000,
                         n_formulas: int = 200,
                         save_intermediates: bool = True) -> Dict[str, Any]:
        """
        Run the complete entanglement analysis pipeline.

        Args:
            sequences: List of symbol sequences for analysis
            n_samples: Number of samples to use for probing
            n_formulas: Number of composition formulas to generate
            save_intermediates: Whether to save intermediate results

        Returns:
            Complete analysis results
        """
        self.logger.info("Starting entanglement analysis pipeline")

        # 1. Data extraction and preprocessing
        self.logger.info("Step 1/8: extracting histories and primitive labels")
        # Convert torch tensors to numpy for data extraction
        sequences_np = [seq.numpy() if isinstance(seq, torch.Tensor) else seq for seq in sequences]
        histories, epsilon_states, primitive_labels = self.data_extractor.get_balanced_sample(
            sequences_np, n_per_state=n_samples // len(self.machine.states)
        )
        self.logger.info(
            "Collected %d histories across %d ε-states (target per-state=%d)",
            len(histories), len(self.machine.states), n_samples // len(self.machine.states)
        )

        # 2. Feature extraction
        self.logger.info("Step 2/8: extracting layer activations from model")
        batch_input = torch.tensor(histories, dtype=torch.long)
        activations = self.feature_extractor.extract_batch(batch_input)
        layer_names = list(activations.keys())

        # 3. Train probes
        self.logger.info("Step 3/8: fitting linear and MLP probes across %d layers", len(layer_names))
        linear_results = self.linear_probes.fit_probes(activations, primitive_labels)
        mlp_results = self.mlp_probes.fit_probes(activations, primitive_labels)
        probe_summary = self.probe_evaluator.summarize_probe_performance(
            linear_results, mlp_results
        )
        if probe_summary:
            top_layer = max(
                probe_summary.items(),
                key=lambda item: item[1]['mean_linear_accuracy']
            )
            self.logger.info(
                "Linear probe mean accuracy best layer %s: %.3f", top_layer[0], top_layer[1]['mean_linear_accuracy']
            )

        # Few-shot evaluation
        linear_few_shot = self.linear_probes.few_shot_evaluation(activations, primitive_labels)
        mlp_few_shot = self.mlp_probes.few_shot_evaluation(activations, primitive_labels)

        # 4. Nonlinearity analysis
        self.logger.info("Step 4/8: computing nonlinearity indices")
        nli_results = self.probe_evaluator.compute_nonlinearity_index(linear_results, mlp_results)

        # 5. Composition evaluation
        self.logger.info("Step 5/8: evaluating compositionality with %d formulas", n_formulas)
        formulas = self.composition_evaluator.generate_formulas(n_formulas)

        # Get logits for composition
        primitive_logits = {}
        for prim_name in primitive_labels.keys():
            prim_logits = self.linear_probes.predict_logits(
                'lm_head_input', prim_name, activations['lm_head_input']
            )
            # Reshape for composition evaluator
            primitive_logits[prim_name] = prim_logits

        composition_results = {}
        for formula in formulas:
            comp_results = self.composition_evaluator.evaluate_formulas(
                [formula], primitive_logits, activations['lm_head_input'], primitive_labels
            )
            composition_results.update(comp_results)

        composition_summary = self.composition_evaluator.aggregate_results(composition_results)
        if composition_summary:
            self.logger.info(
                "Composition zero-shot mean %.3f (n=%d)",
                composition_summary.get('zero_shot_accuracy_mean', 0.0),
                composition_summary.get('n_formulas', 0)
            )

        # 6. Entanglement metrics
        self.logger.info("Step 6/8: computing entanglement metrics")

        # ε-state linearity
        epsilon_linearity = self.entanglement_metrics.compute_epsilon_state_linearity(
            activations['lm_head_input'], epsilon_states
        )

        # Primitive synergy
        primitive_synergy = self.entanglement_metrics.compute_primitive_synergy(
            activations['lm_head_input'], epsilon_states, primitive_labels
        )

        # Fragmentation analysis
        fragmentation_results = {}
        for layer_name, features in activations.items():
            fragmentation_results[layer_name] = self.entanglement_metrics.compute_fragmentation_analysis(
                features, epsilon_states
            )

        # Reachability extrapolation
        reachability_info = self.data_extractor.get_reachability_info()
        extrapolation_results = self.entanglement_metrics.compute_reachability_extrapolation(
            activations, reachability_info, epsilon_states
        )

        # Similarity structure
        similarity_results = self.entanglement_metrics.compute_similarity_structure(
            activations, primitive_labels
        )

        # 7. Trajectory analysis
        self.logger.info("Step 7/8: analyzing trajectory geometry")
        trajectory_results = {}
        for layer_name, features in activations.items():
            # Skip trajectory analysis for now - need proper state sequences
            trajectory_results[layer_name] = {
                'mean_successor_alignment': 0.0,
                'median_successor_alignment': 0.0,
                'std_successor_alignment': 0.0,
                'n_transitions_analyzed': 0,
                'mean_curvature': 0.0,
                'median_curvature': 0.0,
                'std_curvature': 0.0,
                'max_curvature': 0.0,
                'n_windows_analyzed': 0
            }

        # 8. FER score computation
        self.logger.info("Step 8/8: aggregating FER scores")

        # Organize metrics by layer for FER calculation
        layer_metrics = {}
        for layer_name in layer_names:
            layer_metrics[layer_name] = {
                'primitive_probe_summaries': {layer_name: self.probe_evaluator.summarize_probe_performance(
                    {layer_name: linear_results[layer_name]},
                    {layer_name: mlp_results[layer_name]}
                )[layer_name]},
                'nli_results': {layer_name: nli_results[layer_name]},
                'fragmentation_analysis': {layer_name: fragmentation_results[layer_name]},
                'entanglement_metrics': {layer_name: {
                    **epsilon_linearity,
                    **primitive_synergy
                }},
                'trajectory_analysis': {layer_name: trajectory_results[layer_name]}
            }

        fer_scores = self.fer_calculator.compute_layer_wise_fer(layer_metrics)
        best_layer = min(fer_scores.items(), key=lambda item: item[1])
        self.logger.info("Best FER layer %s with score %.3f", best_layer[0], best_layer[1])

        # 9. Compile final results
        results = {
            'metadata': {
                'timestamp': datetime.now().isoformat(),
                'model_path': self.model_path,
                'machine_name': self.machine.name,
                'n_samples': len(histories),
                'n_formulas': n_formulas,
                'layer_names': layer_names,
                'theoretical_limits': {
                    'next_token_bayes_accuracy': self.next_token_bayes_accuracy
                }
            },
            'data_info': {
                'n_epsilon_states': len(self.machine.states),
                'primitive_factors': list(primitive_labels.keys()),
                'sequence_stats': {
                    'total_sequences': len(sequences),
                    'mean_length': np.mean([len(seq) for seq in sequences]),
                    'max_length': max(len(seq) for seq in sequences)
                }
            },
            'probe_results': {
                'linear': linear_results,
                'mlp': mlp_results,
                'few_shot_linear': linear_few_shot,
                'few_shot_mlp': mlp_few_shot,
                'nli_results': nli_results
            },
            'composition_results': {
                'formulas': formulas,
                'formula_results': composition_results,
                'summary': composition_summary
            },
            'entanglement_metrics': {
                'epsilon_linearity': epsilon_linearity,
                'primitive_synergy': primitive_synergy,
                'fragmentation_results': fragmentation_results,
                'extrapolation_results': extrapolation_results,
                'similarity_results': similarity_results
            },
            'trajectory_analysis': trajectory_results,
            'fer_scores': fer_scores,
            'fer_interpretations': {
                layer: self.fer_calculator.get_fer_interpretation(score)
                for layer, score in fer_scores.items()
            }
        }

        # Save results
        if save_intermediates:
            self._save_results(results)

        self.logger.info("Analysis complete")
        return results

    def _save_results(self, results: Dict[str, Any]):
        """Save analysis results to disk."""
        timestamp = results['metadata']['timestamp'].replace(':', '-').replace('.', '-')

        # Save full results as pickle
        results_path = self.output_dir / f"entanglement_analysis_{timestamp}.pkl"
        with open(results_path, 'wb') as f:
            pickle.dump(results, f)

        # Save summary as JSON
        summary = {
            'metadata': results['metadata'],
            'fer_scores': results['fer_scores'],
            'fer_interpretations': results['fer_interpretations'],
            'key_metrics': {
                'best_layer': min(results['fer_scores'], key=results['fer_scores'].get),
                'best_fer_score': min(results['fer_scores'].values()),
                'epsilon_linearity': results['entanglement_metrics']['epsilon_linearity']['accuracy'],
                'composition_success': results['composition_results']['summary'].get('zero_shot_success_rate', 0.0)
            }
        }

        summary_path = self.output_dir / f"entanglement_summary_{timestamp}.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2, default=str)

        self.logger.info("Results saved to %s", results_path)
        self.logger.info("Summary saved to %s", summary_path)

    def generate_report_plots(self, results: Dict[str, Any], save_path: Optional[str] = None):
        """
        Generate the required plots for the analysis report.

        Args:
            results: Complete analysis results
            save_path: Path to save plots (optional)
        """
        # This would implement the plotting functions from the spec:
        # - Linear vs MLP accuracy per label × layer
        # - Zero-/few-shot composition curves
        # - k-extrapolation accuracy vs k
        # - Fragmentation histograms
        # - 2D projections colored by ε-state and primitives
        # - Successor arrows overlay

        # For now, just print that this would be implemented
        print("Report plotting would be implemented here...")
        print("See notebooks/reporting.ipynb for visualization code.")
