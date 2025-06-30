"""
Classical CSSR Analysis Framework

Comprehensive analysis framework for classical CSSR performance
on unified datasets with ground truth evaluation and baseline comparison.
"""

from .classical_analyzer import ClassicalCSSRAnalyzer, BatchAnalyzer
from .dataset_loader import UnifiedDatasetLoader, validate_dataset_for_analysis, DatasetValidationError
from .cssr_runner import CSSRExecutionEngine
from .evaluation_engine import GroundTruthEvaluator, ComparativeEvaluator
from .performance_analyzer import PerformanceAnalyzer
from .results_visualizer import ResultsVisualizer

__all__ = [
    # Main analysis classes
    'ClassicalCSSRAnalyzer',
    'BatchAnalyzer',
    
    # Dataset handling
    'UnifiedDatasetLoader',
    'validate_dataset_for_analysis',
    'DatasetValidationError',
    
    # CSSR execution
    'CSSRExecutionEngine',
    
    # Evaluation and comparison
    'GroundTruthEvaluator',
    'ComparativeEvaluator',
    
    # Performance analysis
    'PerformanceAnalyzer',
    
    # Visualization and reporting
    'ResultsVisualizer',
]

# Version info
__version__ = "1.0.0"
__author__ = "Neural CSSR Project"
__description__ = "Classical CSSR Analysis Framework for Unified Datasets"

# Framework metadata
FRAMEWORK_INFO = {
    'name': 'Classical CSSR Analysis Framework',
    'version': __version__,
    'description': __description__,
    'components': [
        'Dataset Loading and Validation',
        'CSSR Execution with Parameter Sweeps',
        'Ground Truth Evaluation',
        'Performance Analysis and Scaling Studies',
        'Comprehensive Visualization and Reporting',
        'Batch Analysis Across Multiple Datasets'
    ],
    'features': [
        'Parameter sensitivity analysis',
        'Structure recovery evaluation',
        'Baseline performance comparison',
        'Scaling behavior analysis',
        'Interactive HTML reporting',
        'Comprehensive visualizations'
    ]
}


def get_framework_info():
    """Get information about the Classical CSSR Analysis Framework."""
    return FRAMEWORK_INFO


def quick_analysis(dataset_dir: str, output_dir: str, **kwargs):
    """
    Quick analysis function for simple use cases.
    
    Args:
        dataset_dir: Path to unified dataset directory
        output_dir: Path for analysis results
        **kwargs: Additional arguments for ClassicalCSSRAnalyzer
        
    Returns:
        Analysis results dictionary
    """
    analyzer = ClassicalCSSRAnalyzer(dataset_dir, output_dir)
    return analyzer.run_complete_analysis(**kwargs)


def batch_analysis(datasets_dir: str, output_dir: str, dataset_names=None):
    """
    Batch analysis function for analyzing multiple datasets.
    
    Args:
        datasets_dir: Directory containing multiple datasets
        output_dir: Output directory for batch results
        dataset_names: Optional list of specific dataset names
        
    Returns:
        Comparative analysis results
    """
    analyzer = BatchAnalyzer(datasets_dir, output_dir)
    return analyzer.analyze_all_datasets(dataset_names)