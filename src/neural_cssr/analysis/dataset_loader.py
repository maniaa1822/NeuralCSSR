"""
Dataset Loader for Classical CSSR Analysis

Loads and parses unified dataset formats for classical CSSR analysis,
including sequences, ground truth, and statistical metadata.
"""

from pathlib import Path
from typing import Dict, List, Any, Optional
import json


class UnifiedDatasetLoader:
    """Load and parse unified dataset formats for classical CSSR."""
    
    def load_sequences(self, dataset_dir: str, split: str = 'train') -> List[str]:
        """
        Load raw sequences in classical CSSR format.
        
        Args:
            dataset_dir: Unified dataset directory
            split: 'train', 'val', or 'test'
            
        Returns:
            List of sequence strings
        """
        dataset_path = Path(dataset_dir)
        seq_file = dataset_path / 'raw_sequences' / f'{split}_sequences.txt'
        
        if not seq_file.exists():
            raise FileNotFoundError(f"Sequence file not found: {seq_file}")
        
        sequences = []
        with open(seq_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line:  # Skip empty lines
                    sequences.append(line)
        
        if not sequences:
            raise ValueError(f"No sequences found in {seq_file}")
        
        return sequences
    
    def load_ground_truth(self, dataset_dir: str) -> Dict[str, Any]:
        """
        Load ground truth machine definitions and causal states.
        
        Args:
            dataset_dir: Unified dataset directory
            
        Returns:
            Dictionary containing ground truth information:
            {
                'machine_definitions': {...},     # Machine definitions
                'causal_state_labels': {...},     # True state sequences
                'transition_matrices': {...},     # Transition information
                'optimal_predictions': {...}      # Theoretical optima
            }
        """
        dataset_path = Path(dataset_dir)
        gt_dir = dataset_path / 'ground_truth'
        
        if not gt_dir.exists():
            raise FileNotFoundError(f"Ground truth directory not found: {gt_dir}")
        
        ground_truth = {}
        
        # Load all available ground truth files
        gt_files = {
            'machine_definitions.json': 'machine_definitions',
            'causal_state_labels.json': 'causal_state_labels',
            'transition_matrices.json': 'transition_matrices',
            'optimal_predictions.json': 'optimal_predictions',
            'state_trajectories.json': 'state_trajectories',
            'machine_metadata.json': 'machine_metadata'
        }
        
        for filename, key in gt_files.items():
            file_path = gt_dir / filename
            if file_path.exists():
                try:
                    with open(file_path, 'r') as f:
                        ground_truth[key] = json.load(f)
                except json.JSONDecodeError as e:
                    print(f"Warning: Could not parse {filename}: {e}")
                except Exception as e:
                    print(f"Warning: Error loading {filename}: {e}")
        
        if not ground_truth:
            print(f"Warning: No ground truth files found in {gt_dir}")
        
        return ground_truth
    
    def load_statistical_metadata(self, dataset_dir: str) -> Dict[str, Any]:
        """
        Load comprehensive statistical analysis metadata.
        
        Args:
            dataset_dir: Unified dataset directory
            
        Returns:
            Dictionary containing statistical metadata
        """
        dataset_path = Path(dataset_dir)
        stats_dir = dataset_path / 'statistical_analysis'
        
        if not stats_dir.exists():
            print(f"Warning: Statistical analysis directory not found: {stats_dir}")
            return {}
        
        metadata = {}
        
        # Load all available statistical files
        stats_files = {
            'information_measures.json': 'information_measures',
            'complexity_metrics.json': 'complexity_metrics',
            'sequence_statistics.json': 'sequence_statistics',
            'comparative_baselines.json': 'comparative_baselines',
            'entropy_analysis.json': 'entropy_analysis',
            'ngram_analysis.json': 'ngram_analysis'
        }
        
        for filename, key in stats_files.items():
            file_path = stats_dir / filename
            if file_path.exists():
                try:
                    with open(file_path, 'r') as f:
                        metadata[key] = json.load(f)
                except json.JSONDecodeError as e:
                    print(f"Warning: Could not parse {filename}: {e}")
                except Exception as e:
                    print(f"Warning: Error loading {filename}: {e}")
        
        return metadata
    
    def load_experiment_config(self, dataset_dir: str) -> Dict[str, Any]:
        """
        Load experiment configuration used to generate the dataset.
        
        Args:
            dataset_dir: Unified dataset directory
            
        Returns:
            Experiment configuration dictionary
        """
        dataset_path = Path(dataset_dir)
        config_file = dataset_path / 'experiment_config.yaml'
        
        if not config_file.exists():
            # Try JSON format as fallback
            config_file = dataset_path / 'experiment_config.json'
            if config_file.exists():
                with open(config_file, 'r') as f:
                    return json.load(f)
            else:
                print(f"Warning: No experiment config found in {dataset_path}")
                return {}
        
        # Load YAML config
        try:
            import yaml
            with open(config_file, 'r') as f:
                return yaml.safe_load(f)
        except ImportError:
            print("Warning: PyYAML not available, cannot load YAML config")
            return {}
        except Exception as e:
            print(f"Warning: Error loading experiment config: {e}")
            return {}
    
    def load_generation_info(self, dataset_dir: str) -> Dict[str, Any]:
        """
        Load generation information and metadata.
        
        Args:
            dataset_dir: Unified dataset directory
            
        Returns:
            Generation information dictionary
        """
        dataset_path = Path(dataset_dir)
        info_files = ['generation_info.yaml', 'generation_info.json']
        
        for info_file in info_files:
            file_path = dataset_path / info_file
            if file_path.exists():
                try:
                    if info_file.endswith('.yaml'):
                        import yaml
                        with open(file_path, 'r') as f:
                            return yaml.safe_load(f)
                    else:
                        with open(file_path, 'r') as f:
                            return json.load(f)
                except Exception as e:
                    print(f"Warning: Error loading {info_file}: {e}")
                    continue
        
        print(f"Warning: No generation info found in {dataset_path}")
        return {}
    
    def validate_dataset_structure(self, dataset_dir: str) -> Dict[str, bool]:
        """
        Validate that the dataset has the expected unified structure.
        
        Args:
            dataset_dir: Unified dataset directory
            
        Returns:
            Dictionary indicating which components are present
        """
        dataset_path = Path(dataset_dir)
        
        validation = {
            'dataset_dir_exists': dataset_path.exists(),
            'raw_sequences_dir': (dataset_path / 'raw_sequences').exists(),
            'ground_truth_dir': (dataset_path / 'ground_truth').exists(),
            'statistical_analysis_dir': (dataset_path / 'statistical_analysis').exists(),
            'neural_format_dir': (dataset_path / 'neural_format').exists(),
            'quality_reports_dir': (dataset_path / 'quality_reports').exists(),
            'experiment_config': (dataset_path / 'experiment_config.yaml').exists() or 
                                (dataset_path / 'experiment_config.json').exists(),
            'generation_info': (dataset_path / 'generation_info.yaml').exists() or 
                              (dataset_path / 'generation_info.json').exists()
        }
        
        # Check for key sequence files
        if validation['raw_sequences_dir']:
            seq_dir = dataset_path / 'raw_sequences'
            validation['train_sequences'] = (seq_dir / 'train_sequences.txt').exists()
            validation['val_sequences'] = (seq_dir / 'val_sequences.txt').exists()
            validation['test_sequences'] = (seq_dir / 'test_sequences.txt').exists()
        
        # Check for key ground truth files
        if validation['ground_truth_dir']:
            gt_dir = dataset_path / 'ground_truth'
            validation['machine_definitions'] = (gt_dir / 'machine_definitions.json').exists()
            validation['causal_state_labels'] = (gt_dir / 'causal_state_labels.json').exists()
        
        return validation
    
    def get_dataset_summary(self, dataset_dir: str) -> Dict[str, Any]:
        """
        Get comprehensive summary of dataset contents.
        
        Args:
            dataset_dir: Unified dataset directory
            
        Returns:
            Dataset summary with key statistics
        """
        dataset_path = Path(dataset_dir)
        
        summary = {
            'dataset_name': dataset_path.name,
            'dataset_path': str(dataset_path),
            'validation': self.validate_dataset_structure(dataset_dir)
        }
        
        # Add sequence statistics
        try:
            train_sequences = self.load_sequences(dataset_dir, 'train')
            summary['sequence_stats'] = {
                'train_count': len(train_sequences),
                'avg_length': sum(len(seq) for seq in train_sequences) / len(train_sequences),
                'min_length': min(len(seq) for seq in train_sequences),
                'max_length': max(len(seq) for seq in train_sequences)
            }
        except Exception as e:
            summary['sequence_stats'] = {'error': str(e)}
        
        # Add ground truth summary
        try:
            ground_truth = self.load_ground_truth(dataset_dir)
            if 'machine_definitions' in ground_truth:
                machines = ground_truth['machine_definitions']
                summary['ground_truth_stats'] = {
                    'machine_count': len(machines),
                    'state_counts': [len(m.get('states', [])) for m in machines.values() if isinstance(m, dict)]
                }
        except Exception as e:
            summary['ground_truth_stats'] = {'error': str(e)}
        
        # Add experiment config summary
        try:
            config = self.load_experiment_config(dataset_dir)
            if config:
                summary['experiment_config'] = {
                    'experiment_name': config.get('experiment_name', 'unknown'),
                    'machine_specs_count': len(config.get('machine_specs', [])),
                    'generation_config': config.get('generation_config', {})
                }
        except Exception as e:
            summary['experiment_config'] = {'error': str(e)}
        
        return summary


class DatasetValidationError(Exception):
    """Exception raised when dataset validation fails."""
    pass


def validate_dataset_for_analysis(dataset_dir: str) -> bool:
    """
    Validate that a dataset is suitable for classical CSSR analysis.
    
    Args:
        dataset_dir: Path to dataset directory
        
    Returns:
        True if dataset is valid for analysis
        
    Raises:
        DatasetValidationError: If dataset is not suitable for analysis
    """
    loader = UnifiedDatasetLoader()
    validation = loader.validate_dataset_structure(dataset_dir)
    
    # Check required components
    required_components = [
        'dataset_dir_exists',
        'raw_sequences_dir',
        'train_sequences'
    ]
    
    missing_components = []
    for component in required_components:
        if not validation.get(component, False):
            missing_components.append(component)
    
    if missing_components:
        raise DatasetValidationError(
            f"Dataset {dataset_dir} missing required components: {missing_components}"
        )
    
    # Check that sequences can be loaded
    try:
        sequences = loader.load_sequences(dataset_dir, 'train')
        if not sequences:
            raise DatasetValidationError(f"No training sequences found in {dataset_dir}")
    except Exception as e:
        raise DatasetValidationError(f"Error loading sequences from {dataset_dir}: {e}")
    
    return True