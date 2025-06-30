"""
Results Visualizer for Classical CSSR Analysis

Generate comprehensive visualizations and HTML reports
for classical CSSR analysis results.
"""

from typing import Dict, List, Any, Optional
import json
import base64
from io import BytesIO
from pathlib import Path
import numpy as np


class ResultsVisualizer:
    """Generate comprehensive visualizations of classical CSSR analysis."""
    
    def __init__(self):
        """Initialize the visualizer."""
        self.figures = {}
        
    def generate_comprehensive_report(self, analysis_results: Dict, output_dir: str) -> str:
        """
        Generate HTML report with all analysis results and visualizations.
        
        Args:
            analysis_results: Complete analysis results dictionary
            output_dir: Directory to save report and figures
            
        Returns:
            Path to generated HTML report
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        print("Generating visualizations...")
        
        # Create visualizations
        figures = {
            'parameter_heatmap': self._create_parameter_heatmap(analysis_results, output_path),
            'structure_comparison': self._create_structure_comparison(analysis_results, output_path),
            'performance_dashboard': self._create_performance_dashboard(analysis_results, output_path),
            'scaling_analysis': self._create_scaling_plots(analysis_results, output_path),
            'baseline_comparison': self._create_baseline_comparison(analysis_results, output_path),
            'convergence_analysis': self._create_convergence_plots(analysis_results, output_path)
        }
        
        print("Generating HTML report...")
        
        # Generate HTML report
        report_html = self._create_html_report(analysis_results, figures)
        
        # Save report
        report_path = output_path / 'classical_cssr_analysis_report.html'
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_html)
        
        print(f"Report saved to: {report_path}")
        return str(report_path)
    
    def _create_parameter_heatmap(self, results: Dict, output_dir: Path) -> Optional[str]:
        """Create heatmap showing performance across parameter combinations."""
        
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            cssr_results = results.get('cssr_results', {})
            param_results = cssr_results.get('parameter_results', {})
            
            if not param_results:
                return None
            
            # Extract parameter combinations and performance metrics
            data = []
            for param_key, result in param_results.items():
                if 'error' in result:
                    continue
                    
                params = result.get('parameters', {})
                max_length = params.get('max_length', 0)
                significance = params.get('significance_level', 0.0)
                
                num_states = result.get('discovered_structure', {}).get('num_states', 0)
                converged = result.get('execution_info', {}).get('converged', False)
                
                data.append({
                    'max_length': max_length,
                    'significance': significance,
                    'num_states': num_states,
                    'converged': 1 if converged else 0
                })
            
            if not data:
                return None
            
            # Create pivot tables for heatmaps
            import pandas as pd
            df = pd.DataFrame(data)
            
            fig, axes = plt.subplots(1, 2, figsize=(15, 6))
            
            # States heatmap
            states_pivot = df.pivot(index='significance', columns='max_length', values='num_states')
            sns.heatmap(states_pivot, annot=True, fmt='d', cmap='viridis', ax=axes[0])
            axes[0].set_title('Number of Discovered States')
            axes[0].set_xlabel('Maximum History Length')
            axes[0].set_ylabel('Significance Level')
            
            # Convergence heatmap
            conv_pivot = df.pivot(index='significance', columns='max_length', values='converged')
            sns.heatmap(conv_pivot, annot=True, fmt='d', cmap='RdYlGn', ax=axes[1])
            axes[1].set_title('Convergence (1=Converged, 0=Not Converged)')
            axes[1].set_xlabel('Maximum History Length')
            axes[1].set_ylabel('Significance Level')
            
            plt.tight_layout()
            
            # Save figure
            fig_path = output_dir / 'parameter_heatmap.png'
            plt.savefig(fig_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            return str(fig_path)
            
        except ImportError:
            print("Warning: matplotlib/seaborn not available for parameter heatmap")
            return None
        except Exception as e:
            print(f"Error creating parameter heatmap: {e}")
            return None
    
    def _create_structure_comparison(self, results: Dict, output_dir: Path) -> Optional[str]:
        """Visualize discovered vs true causal structures."""
        
        try:
            import matplotlib.pyplot as plt
            
            # Get structure information
            dataset_info = results.get('dataset_info', {})
            cssr_results = results.get('cssr_results', {})
            
            # Get best CSSR result
            best_result = self._get_best_result(cssr_results)
            if not best_result:
                return None
            
            discovered_states = best_result.get('discovered_structure', {}).get('num_states', 0)
            
            # Get true states from ground truth
            gt_machines = dataset_info.get('ground_truth_machines', {})
            true_state_counts = gt_machines.get('state_counts', [])
            
            if not true_state_counts:
                return None
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
            
            # Bar chart comparison
            labels = ['True (Avg)', 'Discovered']
            values = [np.mean(true_state_counts), discovered_states]
            colors = ['skyblue', 'orange']
            
            ax1.bar(labels, values, color=colors)
            ax1.set_ylabel('Number of States')
            ax1.set_title('True vs Discovered States')
            ax1.grid(True, alpha=0.3)
            
            # Add value labels on bars
            for i, v in enumerate(values):
                ax1.text(i, v + 0.1, f'{v:.1f}', ha='center', va='bottom', fontweight='bold')
            
            # Distribution of true state counts
            ax2.hist(true_state_counts, bins=max(1, len(set(true_state_counts))), 
                    alpha=0.7, color='skyblue', edgecolor='black')
            ax2.axvline(discovered_states, color='orange', linestyle='--', linewidth=2, 
                       label=f'Discovered: {discovered_states}')
            ax2.set_xlabel('Number of States')
            ax2.set_ylabel('Frequency')
            ax2.set_title('Distribution of True Machine State Counts')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # Save figure
            fig_path = output_dir / 'structure_comparison.png'
            plt.savefig(fig_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            return str(fig_path)
            
        except ImportError:
            print("Warning: matplotlib not available for structure comparison")
            return None
        except Exception as e:
            print(f"Error creating structure comparison: {e}")
            return None
    
    def _create_performance_dashboard(self, results: Dict, output_dir: Path) -> Optional[str]:
        """Create performance metrics dashboard."""
        
        try:
            import matplotlib.pyplot as plt
            
            eval_metrics = results.get('evaluation_metrics', {})
            baseline_comp = results.get('baseline_comparison', {})
            
            if not eval_metrics and not baseline_comp:
                return None
            
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            fig.suptitle('Performance Dashboard', fontsize=16, fontweight='bold')
            
            # Structure recovery metrics
            structure_recovery = eval_metrics.get('structure_recovery', {})
            if structure_recovery:
                metrics = ['state_count_accuracy', 'transition_recovery_rate', 
                          'history_assignment_accuracy', 'structural_similarity']
                values = [structure_recovery.get(m, 0.0) for m in metrics]
                labels = ['State Count', 'Transitions', 'Histories', 'Overall']
                
                bars = axes[0, 0].bar(labels, values, color='lightcoral')
                axes[0, 0].set_ylim(0, 1)
                axes[0, 0].set_ylabel('Accuracy Score')
                axes[0, 0].set_title('Structure Recovery')
                axes[0, 0].tick_params(axis='x', rotation=45)
                
                # Add value labels
                for bar, val in zip(bars, values):
                    axes[0, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                                   f'{val:.3f}', ha='center', va='bottom')
            
            # Prediction performance
            pred_perf = eval_metrics.get('prediction_performance', {})
            if pred_perf:
                metrics = ['cross_entropy_ratio', 'perplexity_ratio', 'compression_efficiency']
                values = [pred_perf.get(m, 0.0) for m in metrics if pred_perf.get(m, 0.0) > 0]
                labels = ['Cross-Entropy\nRatio', 'Perplexity\nRatio', 'Compression\nEfficiency'][:len(values)]
                
                if values:
                    bars = axes[0, 1].bar(labels, values, color='lightgreen')
                    axes[0, 1].set_ylabel('Ratio/Score')
                    axes[0, 1].set_title('Prediction Performance')
                    axes[0, 1].tick_params(axis='x', rotation=45)
                    
                    # Add value labels
                    for bar, val in zip(bars, values):
                        axes[0, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                                       f'{val:.3f}', ha='center', va='bottom')
            
            # Baseline comparisons
            relative_perf = baseline_comp.get('relative_performance', {})
            if relative_perf and 'component_scores' in relative_perf:
                comp_scores = relative_perf['component_scores']
                if comp_scores:
                    labels = list(comp_scores.keys())
                    values = list(comp_scores.values())
                    
                    bars = axes[1, 0].bar(labels, values, color='lightskyblue')
                    axes[1, 0].set_ylim(0, 1)
                    axes[1, 0].set_ylabel('Performance Score')
                    axes[1, 0].set_title('Baseline Comparisons')
                    axes[1, 0].tick_params(axis='x', rotation=45)
                    
                    # Add value labels
                    for bar, val in zip(bars, values):
                        axes[1, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                                       f'{val:.3f}', ha='center', va='bottom')
            
            # Overall performance summary
            overall_score = relative_perf.get('overall_score', 0.0) if relative_perf else 0.0
            category = relative_perf.get('performance_category', 'unknown') if relative_perf else 'unknown'
            
            # Create a gauge-like visualization
            theta = np.pi * overall_score  # 0 to pi for semicircle
            ax = axes[1, 1]
            ax.set_xlim(-1.1, 1.1)
            ax.set_ylim(-0.1, 1.1)
            
            # Draw semicircle gauge
            angles = np.linspace(0, np.pi, 100)
            x_circle = np.cos(angles)
            y_circle = np.sin(angles)
            ax.plot(x_circle, y_circle, 'k-', linewidth=2)
            
            # Color zones
            colors = ['red', 'orange', 'yellow', 'lightgreen', 'green']
            for i in range(5):
                start_angle = i * np.pi / 5
                end_angle = (i + 1) * np.pi / 5
                angles_zone = np.linspace(start_angle, end_angle, 20)
                ax.fill_between(np.cos(angles_zone), 0, np.sin(angles_zone), 
                               color=colors[i], alpha=0.3)
            
            # Draw needle
            needle_x = np.cos(theta)
            needle_y = np.sin(theta)
            ax.arrow(0, 0, needle_x * 0.8, needle_y * 0.8, 
                    head_width=0.05, head_length=0.05, fc='black', ec='black')
            
            ax.text(0, -0.2, f'Overall Score: {overall_score:.3f}\nCategory: {category.title()}',
                   ha='center', va='center', fontsize=12, fontweight='bold')
            ax.set_title('Overall Performance')
            ax.set_aspect('equal')
            ax.axis('off')
            
            plt.tight_layout()
            
            # Save figure
            fig_path = output_dir / 'performance_dashboard.png'
            plt.savefig(fig_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            return str(fig_path)
            
        except ImportError:
            print("Warning: matplotlib not available for performance dashboard")
            return None
        except Exception as e:
            print(f"Error creating performance dashboard: {e}")
            return None
    
    def _create_scaling_plots(self, results: Dict, output_dir: Path) -> Optional[str]:
        """Create scaling behavior analysis plots."""
        
        try:
            import matplotlib.pyplot as plt
            
            scaling_analysis = results.get('scaling_analysis', {})
            if not scaling_analysis:
                return None
            
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            fig.suptitle('Scaling Analysis', fontsize=16, fontweight='bold')
            
            # Parameter sensitivity
            param_sensitivity = scaling_analysis.get('parameter_sensitivity', {})
            if param_sensitivity:
                self._plot_parameter_sensitivity(param_sensitivity, axes[0, 0])
            
            # Complexity scaling
            complexity_scaling = scaling_analysis.get('complexity_scaling', {})
            if complexity_scaling:
                self._plot_complexity_scaling(complexity_scaling, axes[0, 1])
            
            # Runtime scaling
            runtime_scaling = scaling_analysis.get('runtime_scaling', {})
            if runtime_scaling:
                self._plot_runtime_scaling(runtime_scaling, axes[1, 0])
            
            # Sample efficiency
            sample_scaling = scaling_analysis.get('sample_size_scaling', {})
            if sample_scaling:
                self._plot_sample_scaling(sample_scaling, axes[1, 1])
            
            plt.tight_layout()
            
            # Save figure
            fig_path = output_dir / 'scaling_analysis.png'
            plt.savefig(fig_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            return str(fig_path)
            
        except ImportError:
            print("Warning: matplotlib not available for scaling plots")
            return None
        except Exception as e:
            print(f"Error creating scaling plots: {e}")
            return None
    
    def _create_baseline_comparison(self, results: Dict, output_dir: Path) -> Optional[str]:
        """Create baseline comparison visualizations."""
        
        try:
            import matplotlib.pyplot as plt
            
            baseline_comp = results.get('baseline_comparison', {})
            if not baseline_comp:
                return None
            
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            fig.suptitle('Baseline Comparisons', fontsize=16, fontweight='bold')
            
            # Improvement factors
            improvements = {}
            
            # Random baseline
            vs_random = baseline_comp.get('vs_random_baseline', {})
            if vs_random and 'improvement_factor' in vs_random:
                improvements['Random'] = vs_random['improvement_factor']
            
            # Empirical baselines
            vs_empirical = baseline_comp.get('vs_empirical_baselines', {})
            if vs_empirical:
                for name, comp in vs_empirical.items():
                    if isinstance(comp, dict) and 'improvement_factor' in comp:
                        improvements[f'{name}'] = comp['improvement_factor']
            
            if improvements:
                labels = list(improvements.keys())
                values = list(improvements.values())
                
                bars = axes[0].bar(labels, values, color='lightblue')
                axes[0].set_ylabel('Improvement Factor')
                axes[0].set_title('Improvement Over Baselines')
                axes[0].tick_params(axis='x', rotation=45)
                axes[0].grid(True, alpha=0.3)
                
                # Add value labels
                for bar, val in zip(bars, values):
                    axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                               f'{val:.2f}', ha='center', va='bottom')
            
            # Efficiency vs optimal
            vs_optimal = baseline_comp.get('vs_theoretical_optimal', {})
            if vs_optimal:
                efficiency = vs_optimal.get('efficiency', 0.0)
                
                # Create a simple efficiency meter
                ax = axes[1]
                ax.barh(['Efficiency'], [efficiency], color='green' if efficiency > 0.8 else 'orange')
                ax.set_xlim(0, 1)
                ax.set_xlabel('Efficiency Score')
                ax.set_title('Efficiency vs Theoretical Optimal')
                ax.text(efficiency + 0.02, 0, f'{efficiency:.3f}', 
                       va='center', ha='left', fontweight='bold')
            
            plt.tight_layout()
            
            # Save figure
            fig_path = output_dir / 'baseline_comparison.png'
            plt.savefig(fig_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            return str(fig_path)
            
        except ImportError:
            print("Warning: matplotlib not available for baseline comparison")
            return None
        except Exception as e:
            print(f"Error creating baseline comparison: {e}")
            return None
    
    def _create_convergence_plots(self, results: Dict, output_dir: Path) -> Optional[str]:
        """Create convergence analysis plots."""
        
        try:
            import matplotlib.pyplot as plt
            
            cssr_results = results.get('cssr_results', {})
            convergence_analysis = cssr_results.get('convergence_analysis', {})
            
            if not convergence_analysis:
                return None
            
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            fig.suptitle('Convergence Analysis', fontsize=16, fontweight='bold')
            
            # Convergence by max length
            conv_by_length = convergence_analysis.get('convergence_by_max_length', {})
            if conv_by_length:
                lengths = sorted(conv_by_length.keys())
                rates = [conv_by_length[l].get('rate', 0.0) for l in lengths]
                
                axes[0].plot(lengths, rates, 'o-', linewidth=2, markersize=8)
                axes[0].set_xlabel('Maximum History Length')
                axes[0].set_ylabel('Convergence Rate')
                axes[0].set_title('Convergence vs History Length')
                axes[0].set_ylim(0, 1)
                axes[0].grid(True, alpha=0.3)
                
                # Add value labels
                for x, y in zip(lengths, rates):
                    axes[0].text(x, y + 0.02, f'{y:.2f}', ha='center', va='bottom')
            
            # Convergence by significance level
            conv_by_sig = convergence_analysis.get('convergence_by_significance', {})
            if conv_by_sig:
                sigs = sorted(conv_by_sig.keys())
                rates = [conv_by_sig[s].get('rate', 0.0) for s in sigs]
                
                axes[1].semilogx(sigs, rates, 'o-', linewidth=2, markersize=8)
                axes[1].set_xlabel('Significance Level')
                axes[1].set_ylabel('Convergence Rate')
                axes[1].set_title('Convergence vs Significance Level')
                axes[1].set_ylim(0, 1)
                axes[1].grid(True, alpha=0.3)
                
                # Add value labels
                for x, y in zip(sigs, rates):
                    axes[1].text(x, y + 0.02, f'{y:.2f}', ha='center', va='bottom')
            
            plt.tight_layout()
            
            # Save figure
            fig_path = output_dir / 'convergence_analysis.png'
            plt.savefig(fig_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            return str(fig_path)
            
        except ImportError:
            print("Warning: matplotlib not available for convergence plots")
            return None
        except Exception as e:
            print(f"Error creating convergence plots: {e}")
            return None
    
    def _plot_parameter_sensitivity(self, param_sensitivity: Dict, ax):
        """Plot parameter sensitivity analysis."""
        
        max_length_effects = param_sensitivity.get('max_length_effects', {})
        if max_length_effects:
            lengths = sorted([k for k in max_length_effects.keys() if isinstance(k, (int, float))])
            if lengths:
                mean_states = [max_length_effects[l]['state_count_stats']['mean'] for l in lengths]
                ax.plot(lengths, mean_states, 'o-', linewidth=2, markersize=6)
                ax.set_xlabel('Max History Length')
                ax.set_ylabel('Avg States Discovered')
                ax.set_title('Parameter Sensitivity: Max Length')
                ax.grid(True, alpha=0.3)
    
    def _plot_complexity_scaling(self, complexity_scaling: Dict, ax):
        """Plot complexity scaling analysis."""
        
        disc_vs_true = complexity_scaling.get('discovered_vs_true_states', {})
        if disc_vs_true:
            true_states = disc_vs_true.get('true_states', 0)
            discovered_states = disc_vs_true.get('discovered_states', 0)
            
            ax.bar(['True', 'Discovered'], [true_states, discovered_states], 
                  color=['skyblue', 'orange'])
            ax.set_ylabel('Number of States')
            ax.set_title('Complexity Scaling')
            ax.grid(True, alpha=0.3)
    
    def _plot_runtime_scaling(self, runtime_scaling: Dict, ax):
        """Plot runtime scaling analysis."""
        
        length_scaling = runtime_scaling.get('max_length_scaling', {})
        if length_scaling:
            lengths = length_scaling.get('max_lengths', [])
            runtimes = length_scaling.get('avg_runtimes', [])
            
            if lengths and runtimes:
                ax.plot(lengths, runtimes, 'o-', linewidth=2, markersize=6, color='red')
                ax.set_xlabel('Max History Length')
                ax.set_ylabel('Average Runtime (s)')
                ax.set_title('Runtime Scaling')
                ax.grid(True, alpha=0.3)
    
    def _plot_sample_scaling(self, sample_scaling: Dict, ax):
        """Plot sample efficiency analysis."""
        
        sample_eff = sample_scaling.get('sample_efficiency', {})
        if sample_eff:
            metrics = ['states_per_symbol', 'runtime_per_symbol']
            values = [sample_eff.get(m, 0.0) for m in metrics if sample_eff.get(m, 0.0) > 0]
            labels = ['States/Symbol', 'Runtime/Symbol'][:len(values)]
            
            if values:
                ax.bar(labels, values, color='purple', alpha=0.7)
                ax.set_ylabel('Efficiency Metric')
                ax.set_title('Sample Efficiency')
                ax.tick_params(axis='x', rotation=45)
    
    def _get_best_result(self, cssr_results: Dict) -> Optional[Dict]:
        """Get the best CSSR result from results."""
        
        if 'parameter_results' in cssr_results:
            best_params = cssr_results.get('best_parameters', {})
            overall_best = best_params.get('overall_best', {})
            param_key = overall_best.get('parameter_key')
            
            if param_key and param_key in cssr_results['parameter_results']:
                return cssr_results['parameter_results'][param_key]
        
        elif 'discovered_structure' in cssr_results:
            return cssr_results
        
        return None
    
    def _create_html_report(self, analysis_results: Dict, figures: Dict[str, Optional[str]]) -> str:
        """Create comprehensive HTML report."""
        
        # Basic HTML template
        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Classical CSSR Analysis Report</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            line-height: 1.6;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background-color: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 0 20px rgba(0,0,0,0.1);
        }}
        h1 {{
            color: #2c3e50;
            border-bottom: 3px solid #3498db;
            padding-bottom: 10px;
        }}
        h2 {{
            color: #34495e;
            border-bottom: 2px solid #bdc3c7;
            margin-top: 30px;
            padding-bottom: 5px;
        }}
        h3 {{
            color: #7f8c8d;
            margin-top: 20px;
        }}
        .summary-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }}
        .summary-card {{
            background-color: #ecf0f1;
            padding: 20px;
            border-radius: 8px;
            border-left: 4px solid #3498db;
        }}
        .metric-value {{
            font-size: 1.5em;
            font-weight: bold;
            color: #2c3e50;
        }}
        .metric-label {{
            color: #7f8c8d;
            font-size: 0.9em;
        }}
        .figure {{
            text-align: center;
            margin: 20px 0;
        }}
        .figure img {{
            max-width: 100%;
            height: auto;
            border: 1px solid #ddd;
            border-radius: 8px;
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        }}
        .json-data {{
            background-color: #f8f9fa;
            border: 1px solid #e9ecef;
            border-radius: 5px;
            padding: 15px;
            font-family: 'Courier New', monospace;
            font-size: 0.9em;
            overflow-x: auto;
            max-height: 400px;
            overflow-y: auto;
        }}
        .error {{
            color: #e74c3c;
            background-color: #fdf2f2;
            padding: 10px;
            border-radius: 5px;
            border-left: 4px solid #e74c3c;
        }}
        .success {{
            color: #27ae60;
            background-color: #f2fdf5;
            padding: 10px;
            border-radius: 5px;
            border-left: 4px solid #27ae60;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }}
        th, td {{
            border: 1px solid #ddd;
            padding: 12px;
            text-align: left;
        }}
        th {{
            background-color: #f2f2f2;
            font-weight: bold;
        }}
        .timestamp {{
            color: #7f8c8d;
            font-size: 0.9em;
            margin-top: 20px;
            text-align: right;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Classical CSSR Analysis Report</h1>
        
        {self._generate_summary_section(analysis_results)}
        {self._generate_dataset_info_section(analysis_results)}
        {self._generate_cssr_results_section(analysis_results)}
        {self._generate_evaluation_section(analysis_results)}
        {self._generate_baseline_comparison_section(analysis_results)}
        {self._generate_scaling_analysis_section(analysis_results)}
        {self._generate_figures_section(figures)}
        {self._generate_raw_data_section(analysis_results)}
        
        <div class="timestamp">
            Report generated: {analysis_results.get('analysis_metadata', {}).get('timestamp', 'Unknown')}
        </div>
    </div>
</body>
</html>"""
        
        return html
    
    def _generate_summary_section(self, results: Dict) -> str:
        """Generate executive summary section."""
        
        # Extract key metrics
        dataset_info = results.get('dataset_info', {})
        cssr_results = results.get('cssr_results', {})
        eval_metrics = results.get('evaluation_metrics', {})
        baseline_comp = results.get('baseline_comparison', {})
        
        dataset_name = dataset_info.get('dataset_name', 'Unknown')
        num_sequences = dataset_info.get('num_sequences', 0)
        
        best_result = self._get_best_result(cssr_results)
        discovered_states = best_result.get('discovered_structure', {}).get('num_states', 0) if best_result else 0
        converged = best_result.get('execution_info', {}).get('converged', False) if best_result else False
        
        structure_similarity = eval_metrics.get('structure_recovery', {}).get('structural_similarity', 0.0)
        overall_performance = baseline_comp.get('relative_performance', {}).get('overall_score', 0.0)
        
        return f"""
        <h2>Executive Summary</h2>
        <div class="summary-grid">
            <div class="summary-card">
                <div class="metric-value">{dataset_name}</div>
                <div class="metric-label">Dataset Analyzed</div>
            </div>
            <div class="summary-card">
                <div class="metric-value">{num_sequences:,}</div>
                <div class="metric-label">Training Sequences</div>
            </div>
            <div class="summary-card">
                <div class="metric-value">{discovered_states}</div>
                <div class="metric-label">States Discovered</div>
            </div>
            <div class="summary-card">
                <div class="metric-value">{'Yes' if converged else 'No'}</div>
                <div class="metric-label">CSSR Converged</div>
            </div>
            <div class="summary-card">
                <div class="metric-value">{structure_similarity:.3f}</div>
                <div class="metric-label">Structure Similarity</div>
            </div>
            <div class="summary-card">
                <div class="metric-value">{overall_performance:.3f}</div>
                <div class="metric-label">Overall Performance</div>
            </div>
        </div>
        """
    
    def _generate_dataset_info_section(self, results: Dict) -> str:
        """Generate dataset information section."""
        
        dataset_info = results.get('dataset_info', {})
        
        return f"""
        <h2>Dataset Information</h2>
        <div class="json-data">
{json.dumps(dataset_info, indent=2, default=str)}
        </div>
        """
    
    def _generate_cssr_results_section(self, results: Dict) -> str:
        """Generate CSSR results section."""
        
        cssr_results = results.get('cssr_results', {})
        
        html = "<h2>CSSR Execution Results</h2>"
        
        # Best parameters
        best_params = cssr_results.get('best_parameters', {})
        if best_params:
            html += f"""
            <h3>Best Parameters</h3>
            <div class="json-data">
{json.dumps(best_params, indent=2, default=str)}
            </div>
            """
        
        # Summary statistics
        summary_stats = cssr_results.get('summary_statistics', {})
        if summary_stats:
            html += f"""
            <h3>Summary Statistics</h3>
            <div class="json-data">
{json.dumps(summary_stats, indent=2, default=str)}
            </div>
            """
        
        return html
    
    def _generate_evaluation_section(self, results: Dict) -> str:
        """Generate evaluation metrics section."""
        
        eval_metrics = results.get('evaluation_metrics', {})
        
        if not eval_metrics:
            return "<h2>Evaluation Metrics</h2><p>No evaluation metrics available.</p>"
        
        return f"""
        <h2>Evaluation Metrics</h2>
        <div class="json-data">
{json.dumps(eval_metrics, indent=2, default=str)}
        </div>
        """
    
    def _generate_baseline_comparison_section(self, results: Dict) -> str:
        """Generate baseline comparison section."""
        
        baseline_comp = results.get('baseline_comparison', {})
        
        if not baseline_comp:
            return "<h2>Baseline Comparison</h2><p>No baseline comparison available.</p>"
        
        return f"""
        <h2>Baseline Comparison</h2>
        <div class="json-data">
{json.dumps(baseline_comp, indent=2, default=str)}
        </div>
        """
    
    def _generate_scaling_analysis_section(self, results: Dict) -> str:
        """Generate scaling analysis section."""
        
        scaling_analysis = results.get('scaling_analysis', {})
        
        if not scaling_analysis:
            return "<h2>Scaling Analysis</h2><p>No scaling analysis available.</p>"
        
        return f"""
        <h2>Scaling Analysis</h2>
        <div class="json-data">
{json.dumps(scaling_analysis, indent=2, default=str)}
        </div>
        """
    
    def _generate_figures_section(self, figures: Dict[str, Optional[str]]) -> str:
        """Generate figures section."""
        
        html = "<h2>Visualizations</h2>"
        
        figure_titles = {
            'parameter_heatmap': 'Parameter Heatmap Analysis',
            'structure_comparison': 'Structure Comparison',
            'performance_dashboard': 'Performance Dashboard',
            'scaling_analysis': 'Scaling Analysis',
            'baseline_comparison': 'Baseline Comparison',
            'convergence_analysis': 'Convergence Analysis'
        }
        
        for fig_key, fig_path in figures.items():
            if fig_path and Path(fig_path).exists():
                title = figure_titles.get(fig_key, fig_key.replace('_', ' ').title())
                rel_path = Path(fig_path).name  # Just the filename for HTML
                html += f"""
                <h3>{title}</h3>
                <div class="figure">
                    <img src="{rel_path}" alt="{title}">
                </div>
                """
            else:
                title = figure_titles.get(fig_key, fig_key.replace('_', ' ').title())
                html += f"""
                <h3>{title}</h3>
                <div class="error">Figure could not be generated (matplotlib not available or error occurred)</div>
                """
        
        return html
    
    def _generate_raw_data_section(self, results: Dict) -> str:
        """Generate raw data section."""
        
        return f"""
        <h2>Raw Analysis Data</h2>
        <div class="json-data">
{json.dumps(results, indent=2, default=str)}
        </div>
        """