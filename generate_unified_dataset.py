#!/usr/bin/env python3
"""
Unified Neural CSSR Dataset Generation Script

This script generates datasets using the new unified framework with
comprehensive metadata, quality validation, and multiple output formats.
Supports both enumeration-based generation (many short sequences from multiple machines)
and single machine generation (few very long sequences from specific machines).

Usage Examples:

  # Enumeration-based generation (standard datasets)
  python generate_unified_dataset.py --preset small --output datasets/small_exp
  python generate_unified_dataset.py --preset medium --output datasets/medium_exp
  python generate_unified_dataset.py --preset large --output datasets/large_exp
  python generate_unified_dataset.py --preset biased --output datasets/biased_exp
  
  # Single machine generation (individual machine analysis)
  python generate_unified_dataset.py --preset transcssr --output datasets/single_machines
  
  # Custom configuration
  python generate_unified_dataset.py --config custom_config.yaml --output datasets/custom_exp
  
  # Preview configuration without generating
  python generate_unified_dataset.py --preset medium --output /tmp --dry-run

Available Presets:
  - small: 5 machines, 2.5K sequences, uniform probabilities (fast testing)
  - medium: 16 machines, 25K sequences, uniform probabilities (standard research)  
  - large: 33 machines, 122K sequences, uniform probabilities (comprehensive studies)
  - biased: 6 machines, 6.5K sequences, mixed uniform/biased probabilities (bias studies)
  - transcssr: 6 machines, long sequences (50K-75K), single machine generation mode

Single Machine Generation:
  The 'transcssr' preset generates individual machines without enumeration:
  - One very long sequence per machine (50,000-75,000 symbols)
  - Steady-state generation with burn-in period
  - Compatible with classical CSSR tools like transCSSR
  - Ideal for individual machine analysis and classical CSSR comparison
"""

import argparse
import sys
import time
from pathlib import Path
from typing import Dict, Any

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

try:
    from neural_cssr.data.dataset_generator import UnifiedDatasetGenerator
    from neural_cssr.config.generation_schemas import create_example_configs, load_config_from_dict
    import yaml
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure you're running from the project root directory")
    sys.exit(1)


def create_preset_configs() -> Dict[str, Dict[str, Any]]:
    """Create preset configurations for common experiments."""
    
    presets = {
        'small': {
            'experiment_name': 'small_neural_cssr_experiment',
            'random_seed': 42,
            'machine_specs': [
                {
                    'complexity_class': '2-state-binary',
                    'machine_count': 3,
                    'samples_per_machine': 500,
                    'weight': 1.0
                },
                {
                    'complexity_class': '3-state-binary',
                    'machine_count': 2,
                    'samples_per_machine': 500,
                    'weight': 1.0
                }
            ],
            'sequence_spec': {
                'length_distribution': [20, 100],
                'total_sequences': 2500,
                'train_ratio': 0.7,
                'val_ratio': 0.15,
                'test_ratio': 0.15
            },
            'quality_spec': {
                'min_state_coverage': 30,
                'min_transition_coverage': 15,
                'entropy_tolerance': 0.05,
                'length_diversity_threshold': 0.3
            },
            'neural_format_config': {
                'context_length': 256,
                'vocab_special_tokens': ['<PAD>', '<UNK>'],
                'include_position_metadata': True,
                'batch_friendly_padding': True
            },
            'output_config': {
                'save_raw_sequences': True,
                'save_neural_format': True,
                'save_statistical_analysis': True,
                'save_quality_reports': True,
                'compress_outputs': False
            }
        },
        
        'medium': {
            'experiment_name': 'medium_neural_cssr_experiment',
            'random_seed': 42,
            'machine_specs': [
                {
                    'complexity_class': '2-state-binary',
                    'machine_count': 5,
                    'samples_per_machine': 2000,
                    'weight': 1.0
                },
                {
                    'complexity_class': '3-state-binary',
                    'machine_count': 8,
                    'samples_per_machine': 1500,
                    'weight': 1.2
                },
                {
                    'complexity_class': '4-state-binary',
                    'machine_count': 3,
                    'samples_per_machine': 1000,
                    'weight': 0.8
                }
            ],
            'sequence_spec': {
                'length_distribution': [50, 300],
                'total_sequences': 25000,
                'train_ratio': 0.7,
                'val_ratio': 0.15,
                'test_ratio': 0.15
            },
            'quality_spec': {
                'min_state_coverage': 100,
                'min_transition_coverage': 50,
                'entropy_tolerance': 0.03,
                'length_diversity_threshold': 0.4
            },
            'neural_format_config': {
                'context_length': 512,
                'vocab_special_tokens': ['<PAD>', '<UNK>'],
                'include_position_metadata': True,
                'batch_friendly_padding': True
            },
            'output_config': {
                'save_raw_sequences': True,
                'save_neural_format': True,
                'save_statistical_analysis': True,
                'save_quality_reports': True,
                'compress_outputs': False
            }
        },
        
        'biased': {
            'experiment_name': 'biased_neural_cssr_experiment',
            'random_seed': 42,
            'machine_specs': [
                {
                    'complexity_class': '2-state-binary',
                    'machine_count': 2,
                    'samples_per_machine': 1000,
                    'weight': 1.0,
                    'topological': True
                },
                {
                    'complexity_class': '2-state-binary',
                    'machine_count': 3,
                    'samples_per_machine': 1000,
                    'weight': 1.2,
                    'topological': False,
                    'bias_strength': 0.7,
                    'probability_seed': 123
                },
                {
                    'complexity_class': '3-state-binary',
                    'machine_count': 1,
                    'samples_per_machine': 1500,
                    'weight': 1.5,
                    'topological': False,
                    'custom_probabilities': {
                        'S0': {'0': 0.8, '1': 0.2},
                        'S1': {'0': 0.3, '1': 0.7},
                        'S2': {'0': 0.5, '1': 0.5}
                    }
                }
            ],
            'sequence_spec': {
                'length_distribution': [30, 150],
                'total_sequences': 6500,
                'train_ratio': 0.7,
                'val_ratio': 0.15,
                'test_ratio': 0.15
            },
            'quality_spec': {
                'min_state_coverage': 50,
                'min_transition_coverage': 25,
                'entropy_tolerance': 0.1,
                'length_diversity_threshold': 0.3
            },
            'neural_format_config': {
                'context_length': 256,
                'vocab_special_tokens': ['<PAD>', '<UNK>'],
                'include_position_metadata': True,
                'batch_friendly_padding': True
            },
            'output_config': {
                'save_raw_sequences': True,
                'save_neural_format': True,
                'save_statistical_analysis': True,
                'save_quality_reports': True,
                'compress_outputs': False
            }
        },
        
        'large': {
            'experiment_name': 'large_neural_cssr_experiment',
            'random_seed': 42,
            'machine_specs': [
                {
                    'complexity_class': '2-state-binary',
                    'machine_count': 10,
                    'samples_per_machine': 5000,
                    'weight': 1.0
                },
                {
                    'complexity_class': '3-state-binary',
                    'machine_count': 15,
                    'samples_per_machine': 4000,
                    'weight': 1.1
                },
                {
                    'complexity_class': '4-state-binary',
                    'machine_count': 8,
                    'samples_per_machine': 3000,
                    'weight': 1.3
                }
            ],
            'sequence_spec': {
                'length_distribution': [100, 800],
                'total_sequences': 122000,
                'train_ratio': 0.7,
                'val_ratio': 0.15,
                'test_ratio': 0.15
            },
            'quality_spec': {
                'min_state_coverage': 200,
                'min_transition_coverage': 100,
                'entropy_tolerance': 0.02,
                'length_diversity_threshold': 0.5
            },
            'neural_format_config': {
                'context_length': 1024,
                'vocab_special_tokens': ['<PAD>', '<UNK>'],
                'include_position_metadata': True,
                'batch_friendly_padding': True
            },
            'output_config': {
                'save_raw_sequences': True,
                'save_neural_format': True,
                'save_statistical_analysis': True,
                'save_quality_reports': True,
                'compress_outputs': True
            }
        },
        
        'transcssr': {
            'experiment_name': 'transcssr_compatible_biased',
            'random_seed': 42,
            'machine_specs': [
                {
                    'complexity_class': '2-state-binary',
                    'machine_count': 2,
                    'samples_per_machine': 1,
                    'weight': 1.0,
                    'topological': False,
                    'bias_strength': 0.7,
                    'probability_seed': 42
                },
                {
                    'complexity_class': '3-state-binary',
                    'machine_count': 3,
                    'samples_per_machine': 1,
                    'weight': 1.0,
                    'topological': False,
                    'custom_probabilities': {
                        'S0': {'0': 0.8, '1': 0.2},
                        'S1': {'0': 0.3, '1': 0.7},
                        'S2': {'0': 0.5, '1': 0.5}
                    }
                },
                {
                    'complexity_class': '2-state-binary',
                    'machine_count': 1,
                    'samples_per_machine': 1,
                    'weight': 1.0,
                    'topological': True
                }
            ],
            'sequence_spec': {
                'length_distribution': [50000, 75000],
                'total_sequences': 6,
                'train_ratio': 0.7,
                'val_ratio': 0.15,
                'test_ratio': 0.15,
                'use_steady_state': True,
                'burn_in_length': 20,
                'contiguous_mode': True
            },
            'quality_spec': {
                'min_state_coverage': 100,
                'min_transition_coverage': 50,
                'entropy_tolerance': 0.05,
                'length_diversity_threshold': 0.2
            },
            'neural_format_config': {
                'context_length': 512,
                'vocab_special_tokens': ['<PAD>', '<UNK>'],
                'include_position_metadata': True,
                'batch_friendly_padding': True
            },
            'output_config': {
                'save_raw_sequences': True,
                'save_neural_format': True,
                'save_statistical_analysis': True,
                'save_quality_reports': True,
                'compress_outputs': False
            }
        }
    }
    
    return presets


def load_config(config_path: str = None, preset: str = None) -> Dict[str, Any]:
    """Load configuration from file or preset."""
    
    if preset:
        presets = create_preset_configs()
        if preset not in presets:
            raise ValueError(f"Unknown preset '{preset}'. Available: {list(presets.keys())}")
        return presets[preset]
        
    elif config_path:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
            
    else:
        raise ValueError("Must specify either --config or --preset")


def print_config_summary(config: Dict[str, Any]):
    """Print a summary of the configuration."""
    print(f"\n📋 Experiment: {config['experiment_name']}")
    print(f"🎲 Random seed: {config['random_seed']}")
    
    print(f"\n🔧 Machine Specifications:")
    total_machines = 0
    total_sequences = 0
    for spec in config['machine_specs']:
        count = spec['machine_count']
        samples = spec['samples_per_machine']
        total_machines += count
        total_sequences += count * samples
        print(f"  • {spec['complexity_class']}: {count} machines × {samples} sequences (weight: {spec['weight']})")
    
    print(f"\n📊 Dataset Summary:")
    print(f"  • Total machines: {total_machines}")
    print(f"  • Total sequences: {total_sequences}")
    
    seq_spec = config['sequence_spec']
    print(f"  • Sequence lengths: {seq_spec['length_distribution'][0]}-{seq_spec['length_distribution'][1]}")
    print(f"  • Train/Val/Test split: {seq_spec['train_ratio']:.1%}/{seq_spec['val_ratio']:.1%}/{seq_spec['test_ratio']:.1%}")
    
    neural_config = config['neural_format_config']
    print(f"\n🧠 Neural Format:")
    print(f"  • Context length: {neural_config['context_length']}")
    print(f"  • Special tokens: {neural_config['vocab_special_tokens']}")


def print_generation_progress(phase: str, details: str = ""):
    """Print progress information."""
    timestamp = time.strftime("%H:%M:%S")
    print(f"[{timestamp}] {phase}")
    if details:
        print(f"           {details}")


def print_generation_report(report: Dict[str, Any]):
    """Print comprehensive generation report."""
    print(f"\n✅ Dataset Generation Complete!")
    print(f"=" * 50)
    
    print(f"📁 Output Directory: {report['output_directory']}")
    print(f"🔬 Experiment: {report['experiment_name']}")
    print(f"📈 Quality Score: {report['quality_score']:.3f}/1.0")
    
    print(f"\n📊 Dataset Statistics:")
    print(f"  • Total sequences: {report['total_sequences']:,}")
    print(f"  • Total machines: {report['total_machines']}")
    
    print(f"\n📂 Data Splits:")
    for split, size in report['split_sizes'].items():
        percentage = (size / report['total_sequences']) * 100
        print(f"  • {split.capitalize()}: {size:,} ({percentage:.1f}%)")
    
    print(f"\n🎯 Next Steps:")
    print(f"  • Raw sequences: {report['output_directory']}/raw_sequences/")
    print(f"  • Neural datasets: {report['output_directory']}/neural_format/")
    print(f"  • Ground truth: {report['output_directory']}/ground_truth/")
    print(f"  • Quality reports: {report['output_directory']}/quality_reports/")


def save_generation_script(output_dir: Path, config: Dict[str, Any], args):
    """Save the generation script and parameters for reproducibility."""
    script_info = {
        'generation_script': str(Path(__file__).name),
        'command_line_args': vars(args),
        'config_used': config,
        'generation_timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
        'framework_version': 'unified_v1.0'
    }
    
    with open(output_dir / 'generation_info.yaml', 'w') as f:
        yaml.dump(script_info, f, indent=2)


def main():
    """Main dataset generation function."""
    parser = argparse.ArgumentParser(description='Generate Neural CSSR datasets using unified framework')
    
    # Configuration options
    config_group = parser.add_mutually_exclusive_group(required=True)
    config_group.add_argument('--config', '-c', 
                             help='Path to YAML configuration file')
    config_group.add_argument('--preset', '-p', 
                             choices=['small', 'medium', 'large', 'biased', 'transcssr'],
                             help='Use predefined configuration preset')
    
    # Output options
    parser.add_argument('--output', '-o', required=True,
                       help='Output directory for generated dataset')
    parser.add_argument('--seed', type=int,
                       help='Random seed (overrides config)')
    parser.add_argument('--name', 
                       help='Experiment name (overrides config)')
    
    # Behavior options
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Verbose output')
    parser.add_argument('--dry-run', action='store_true',
                       help='Show configuration without generating dataset')
    
    args = parser.parse_args()
    
    try:
        # Load configuration
        print_generation_progress("Loading configuration...")
        config = load_config(args.config, args.preset)
        
        # Apply command line overrides
        if args.seed is not None:
            config['random_seed'] = args.seed
        if args.name:
            config['experiment_name'] = args.name
            
        # Print configuration summary
        print_config_summary(config)
        
        if args.dry_run:
            print(f"\n🔍 Dry run complete - configuration loaded successfully")
            return
            
        # Create output directory
        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save configuration and generation info
        print_generation_progress("Saving configuration...")
        save_generation_script(output_dir, config, args)
        
        # Create temporary config file for generator
        temp_config_path = output_dir / 'temp_config.yaml'
        with open(temp_config_path, 'w') as f:
            yaml.dump(config, f, indent=2)
        
        # Initialize dataset generator
        print_generation_progress("Initializing dataset generator...")
        generator = UnifiedDatasetGenerator(
            config_path=str(temp_config_path),
            output_dir=str(output_dir),
            seed=config['random_seed']
        )
        
        # Generate dataset
        print_generation_progress("Starting dataset generation...")
        start_time = time.time()
        
        report = generator.generate_dataset()
        
        generation_time = time.time() - start_time
        report['generation_time_seconds'] = generation_time
        
        # Clean up temporary config
        temp_config_path.unlink()
        
        # Print final report
        print_generation_report(report)
        print(f"\n⏱️  Generation time: {generation_time:.1f} seconds")
        print(f"\n🎉 Dataset ready for training and analysis!")
        
    except KeyboardInterrupt:
        print(f"\n\n⚠️  Generation interrupted by user")
        sys.exit(1)
        
    except Exception as e:
        print(f"\n\n❌ Error during dataset generation:")
        print(f"   {type(e).__name__}: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()