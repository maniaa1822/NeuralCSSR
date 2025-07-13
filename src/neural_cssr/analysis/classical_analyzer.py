"""
Classical CSSR Analysis Framework - Main Analyzer Class

Comprehensive classical CSSR analysis for unified datasets with
end-to-end analysis pipeline from dataset loading to evaluation.
"""

from pathlib import Path
from typing import Dict, Any, List, Optional
import json
import time
from datetime import datetime

from .dataset_loader import UnifiedDatasetLoader
from .cssr_runner import CSSRExecutionEngine
from .evaluation_engine import GroundTruthEvaluator
from .performance_analyzer import PerformanceAnalyzer
from .results_visualizer import ResultsVisualizer

# Import machine distance analysis
try:
    from ..evaluation.machine_distance import MachineDistanceCalculator
    from ..evaluation.utils.data_loading import load_ground_truth
    DISTANCE_ANALYSIS_AVAILABLE = True
except ImportError:
    DISTANCE_ANALYSIS_AVAILABLE = False


class ClassicalCSSRAnalyzer:
    """
    Comprehensive classical CSSR analysis for unified datasets.
    
    Provides end-to-end analysis pipeline from dataset loading
    to comprehensive evaluation against ground truth.
    """
    
    def __init__(self, dataset_dir: str, output_dir: str):
        """
        Initialize analyzer with dataset and output directories.
        
        Args:
            dataset_dir: Path to unified dataset directory
            output_dir: Path for analysis results
        """
        self.dataset_dir = Path(dataset_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.dataset_metadata = None
        self.ground_truth = None
        self.cssr_results = None
        
        # Initialize components
        self.dataset_loader = UnifiedDatasetLoader()
        self.performance_analyzer = PerformanceAnalyzer()
        self.results_visualizer = ResultsVisualizer()
        
    def run_complete_analysis(self, 
                            max_length: int = 10, 
                            significance_level: float = 0.05,
                            parameter_sweep: bool = True,
                            distance_analysis: bool = False) -> Dict[str, Any]:
        """
        Run complete classical CSSR analysis pipeline.
        
        Args:
            max_length: Maximum history length for CSSR
            significance_level: Statistical significance level
            parameter_sweep: Whether to run parameter sweep analysis
            distance_analysis: Whether to run machine distance analysis
        
        Returns:
            comprehensive_results: All analysis results and metrics
        """
        print(f"Starting Classical CSSR Analysis for dataset: {self.dataset_dir.name}")
        start_time = time.time()
        
        # Step 1: Load dataset and metadata
        print("Step 1: Loading dataset and metadata...")
        self.load_dataset()
        
        # Step 2: Run classical CSSR
        print("Step 2: Running classical CSSR...")
        if parameter_sweep:
            self.run_classical_cssr_sweep()
        else:
            self.run_classical_cssr_single(max_length, significance_level)
        
        # Step 3: Evaluate against ground truth
        print("Step 3: Evaluating against ground truth...")
        evaluation_metrics = self.evaluate_against_ground_truth()
        
        # Step 4: Compute performance baselines
        print("Step 4: Computing baseline comparisons...")
        baseline_comparison = self.compare_against_baselines()
        
        # Step 5: Analyze scaling behavior
        print("Step 5: Analyzing scaling behavior...")
        scaling_analysis = self.analyze_scaling_behavior()
        
        # Step 6: Machine distance analysis (optional)
        distance_results = None
        if distance_analysis:
            print("Step 6: Running machine distance analysis...")
            distance_results = self.run_machine_distance_analysis()
        
        # Step 7: Generate comprehensive report
        print(f"Step {'7' if distance_analysis else '6'}: Generating analysis report...")
        results = self.generate_analysis_report(
            evaluation_metrics, baseline_comparison, scaling_analysis, distance_results
        )
        
        total_time = time.time() - start_time
        results['analysis_metadata'] = {
            'total_analysis_time': total_time,
            'timestamp': datetime.now().isoformat(),
            'dataset_dir': str(self.dataset_dir),
            'output_dir': str(self.output_dir)
        }
        
        print(f"Analysis complete in {total_time:.2f} seconds")
        return results
    
    def load_dataset(self):
        """Load dataset sequences, ground truth, and metadata."""
        # Load sequences
        self.sequences = self.dataset_loader.load_sequences(self.dataset_dir, 'train')
        print(f"Loaded {len(self.sequences)} sequences")
        
        # Load ground truth
        self.ground_truth = self.dataset_loader.load_ground_truth(self.dataset_dir)
        print(f"Loaded ground truth for {len(self.ground_truth.get('machine_definitions', {}))} machines")
        
        # Load statistical metadata
        self.dataset_metadata = self.dataset_loader.load_statistical_metadata(self.dataset_dir)
        print(f"Loaded statistical metadata with {len(self.dataset_metadata)} categories")
    
    def run_classical_cssr_single(self, max_length: int, significance_level: float):
        """Run CSSR with single parameter set."""
        engine = CSSRExecutionEngine(self.sequences)
        self.cssr_results = engine.run_single_analysis(max_length, significance_level)
    
    def run_classical_cssr_sweep(self):
        """Run CSSR parameter sweep analysis."""
        engine = CSSRExecutionEngine(self.sequences)
        self.cssr_results = engine.run_parameter_sweep()
    
    def evaluate_against_ground_truth(self) -> Dict[str, Any]:
        """Evaluate CSSR results against ground truth."""
        if not self.ground_truth or not self.cssr_results:
            return {'error': 'Missing ground truth or CSSR results'}
        
        evaluator = GroundTruthEvaluator(self.cssr_results, self.ground_truth)
        
        evaluation = {
            'structure_recovery': evaluator.evaluate_structure_recovery(),
            'prediction_performance': evaluator.evaluate_prediction_performance(),
        }
        
        return evaluation
    
    def compare_against_baselines(self) -> Dict[str, Any]:
        """Compare CSSR performance against baselines."""
        if not self.cssr_results or not self.dataset_metadata:
            return {'error': 'Missing CSSR results or metadata'}
        
        return self.performance_analyzer.compare_against_baselines(
            self.cssr_results, self.dataset_metadata
        )
    
    def analyze_scaling_behavior(self) -> Dict[str, Any]:
        """Analyze CSSR scaling behavior."""
        if not self.cssr_results or not self.dataset_metadata:
            return {'error': 'Missing CSSR results or metadata'}
        
        return self.performance_analyzer.analyze_scaling_behavior(
            self.cssr_results, self.dataset_metadata
        )
    
    def generate_analysis_report(self, 
                               evaluation_metrics: Dict, 
                               baseline_comparison: Dict, 
                               scaling_analysis: Dict,
                               distance_results: Optional[Dict] = None) -> Dict[str, Any]:
        """Generate comprehensive analysis report."""
        
        results = {
            'dataset_info': self._extract_dataset_info(),
            'cssr_results': self.cssr_results,
            'evaluation_metrics': evaluation_metrics,
            'baseline_comparison': baseline_comparison,
            'scaling_analysis': scaling_analysis
        }
        
        # Add distance analysis results if available
        if distance_results is not None:
            results['machine_distance_analysis'] = distance_results
        
        # Save JSON results
        results_file = self.output_dir / 'classical_cssr_results.json'
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        # Generate HTML report
        html_report = self.results_visualizer.generate_comprehensive_report(
            results, str(self.output_dir)
        )
        
        results['report_files'] = {
            'json_results': str(results_file),
            'html_report': html_report
        }
        
        return results
    
    def run_machine_distance_analysis(self) -> Optional[Dict[str, Any]]:
        """
        Run machine distance analysis comparing CSSR results to ground truth.
        
        Returns:
            Distance analysis results or None if analysis unavailable
        """
        if not DISTANCE_ANALYSIS_AVAILABLE:
            print("Warning: Machine distance analysis not available (missing dependencies)")
            return None
        
        if not self.cssr_results:
            print("Warning: No CSSR results available for distance analysis")
            return None
            
        try:
            # Load ground truth in the format expected by distance analysis
            print("  Loading ground truth for distance analysis...")
            ground_truth_machines = load_ground_truth(str(self.dataset_dir))
            
            # Create distance calculator
            calculator = MachineDistanceCalculator()
            
            # Compute distances
            print("  Computing distance metrics...")
            distance_results = calculator.compute_all_distances(
                self.cssr_results, ground_truth_machines
            )
            
            # Generate distance analysis report
            distance_output_dir = self.output_dir / 'machine_distance_analysis'
            distance_output_dir.mkdir(exist_ok=True)
            
            print("  Generating distance analysis report...")
            report_path = calculator.generate_report(
                distance_results, 
                str(distance_output_dir / 'machine_distance_report.md')
            )
            
            # Generate visualizations
            try:
                from ..evaluation.utils.visualization import create_distance_visualizations
                viz_files = create_distance_visualizations(distance_results, str(distance_output_dir))
                distance_results['visualization_files'] = viz_files
            except Exception as e:
                print(f"  Warning: Could not generate distance visualizations: {e}")
            
            distance_results['report_path'] = report_path
            
            # Print summary
            summary = distance_results.get('summary', {})
            print(f"  ✓ Distance analysis complete:")
            print(f"    Overall Quality: {summary.get('overall_quality_score', 0.0):.3f}")
            print(f"    Confidence: {summary.get('confidence', 0.0):.3f}")
            print(f"    Best Metric: {summary.get('best_metric', 'Unknown')}")
            
            return distance_results
            
        except Exception as e:
            print(f"Warning: Machine distance analysis failed: {e}")
            return None
    
    def _extract_dataset_info(self) -> Dict[str, Any]:
        """Extract key dataset information for reporting."""
        info = {
            'dataset_name': self.dataset_dir.name,
            'num_sequences': len(self.sequences) if hasattr(self, 'sequences') else 0,
        }
        
        # Add metadata if available
        if self.dataset_metadata:
            if 'sequence_statistics' in self.dataset_metadata:
                seq_stats = self.dataset_metadata['sequence_statistics']
                info.update({
                    'sequence_length_stats': seq_stats.get('length_statistics', {}),
                    'symbol_distribution': seq_stats.get('symbol_counts', {})
                })
            
            if 'complexity_metrics' in self.dataset_metadata:
                complexity = self.dataset_metadata['complexity_metrics']
                info['complexity_metrics'] = complexity
        
        # Add ground truth info
        if self.ground_truth and 'machine_definitions' in self.ground_truth:
            machines = self.ground_truth['machine_definitions']
            info['ground_truth_machines'] = {
                'count': len(machines),
                'state_counts': [len(m.get('states', [])) for m in machines.values()]
            }
        
        return info


class BatchAnalyzer:
    """Batch analysis across multiple datasets."""
    
    def __init__(self, datasets_dir: str, output_dir: str):
        self.datasets_dir = Path(datasets_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def analyze_all_datasets(self, dataset_names: Optional[List[str]] = None, distance_analysis: bool = False) -> Dict[str, Any]:
        """
        Analyze all datasets in the datasets directory.
        
        Args:
            dataset_names: Specific datasets to analyze, or None for all
            distance_analysis: Whether to run machine distance analysis
            
        Returns:
            Comparative analysis results across all datasets
        """
        if dataset_names is None:
            dataset_names = [d.name for d in self.datasets_dir.iterdir() if d.is_dir()]
        
        results = {}
        
        for dataset_name in dataset_names:
            dataset_dir = self.datasets_dir / dataset_name
            if not dataset_dir.exists():
                print(f"Warning: Dataset {dataset_name} not found")
                continue
            
            print(f"\nAnalyzing dataset: {dataset_name}")
            
            analyzer = ClassicalCSSRAnalyzer(
                dataset_dir=str(dataset_dir),
                output_dir=str(self.output_dir / dataset_name)
            )
            
            try:
                dataset_results = analyzer.run_complete_analysis(distance_analysis=distance_analysis)
                results[dataset_name] = dataset_results
            except Exception as e:
                print(f"Error analyzing {dataset_name}: {e}")
                results[dataset_name] = {'error': str(e)}
        
        # Generate comparative report
        comparative_results = self._generate_comparative_analysis(results)
        
        # Save comparative results
        comp_file = self.output_dir / 'comparative_analysis.json'
        with open(comp_file, 'w') as f:
            json.dump(comparative_results, f, indent=2, default=str)
        
        return comparative_results
    
    def _generate_comparative_analysis(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comparative analysis across datasets."""
        
        successful_results = {k: v for k, v in results.items() if 'error' not in v}
        
        if not successful_results:
            return {'error': 'No successful analyses to compare'}
        
        comparative = {
            'summary': {
                'total_datasets': len(results),
                'successful_analyses': len(successful_results),
                'failed_analyses': len(results) - len(successful_results)
            },
            'dataset_comparison': {},
            'aggregate_metrics': {}
        }
        
        # Dataset-by-dataset comparison
        for dataset, result in successful_results.items():
            comparative['dataset_comparison'][dataset] = {
                'dataset_info': result.get('dataset_info', {}),
                'best_cssr_params': self._extract_best_params(result),
                'structure_recovery_score': self._extract_structure_score(result),
                'prediction_performance': self._extract_prediction_performance(result)
            }
        
        # Aggregate metrics
        comparative['aggregate_metrics'] = self._compute_aggregate_metrics(successful_results)
        
        return comparative
    
    def _extract_best_params(self, result: Dict) -> Dict:
        """Extract best CSSR parameters from result."""
        cssr_results = result.get('cssr_results', {})
        return cssr_results.get('best_parameters', {})
    
    def _extract_structure_score(self, result: Dict) -> float:
        """Extract structure recovery score."""
        eval_metrics = result.get('evaluation_metrics', {})
        structure = eval_metrics.get('structure_recovery', {})
        return structure.get('state_count_accuracy', 0.0)
    
    def _extract_prediction_performance(self, result: Dict) -> Dict:
        """Extract prediction performance metrics."""
        eval_metrics = result.get('evaluation_metrics', {})
        return eval_metrics.get('prediction_performance', {})
    
    def _compute_aggregate_metrics(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Compute aggregate metrics across datasets."""
        
        structure_scores = []
        prediction_ratios = []
        
        for result in results.values():
            structure_score = self._extract_structure_score(result)
            if structure_score > 0:
                structure_scores.append(structure_score)
            
            pred_perf = self._extract_prediction_performance(result)
            if 'cross_entropy_ratio' in pred_perf:
                prediction_ratios.append(pred_perf['cross_entropy_ratio'])
        
        aggregate = {}
        
        if structure_scores:
            aggregate['structure_recovery'] = {
                'mean': sum(structure_scores) / len(structure_scores),
                'min': min(structure_scores),
                'max': max(structure_scores),
                'count': len(structure_scores)
            }
        
        if prediction_ratios:
            aggregate['prediction_performance'] = {
                'mean_cross_entropy_ratio': sum(prediction_ratios) / len(prediction_ratios),
                'best_cross_entropy_ratio': min(prediction_ratios),
                'worst_cross_entropy_ratio': max(prediction_ratios),
                'count': len(prediction_ratios)
            }
        
        return aggregate