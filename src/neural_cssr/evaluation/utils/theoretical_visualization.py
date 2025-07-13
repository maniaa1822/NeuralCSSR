"""Visualization utilities for theoretical epsilon machine analysis."""

import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Any, Optional
import seaborn as sns
from pathlib import Path


def create_theoretical_visualizations(results: Dict[str, Any], output_dir: str) -> List[str]:
    """
    Create comprehensive visualizations for theoretical epsilon machine analysis.
    
    Args:
        results: Results from comprehensive analysis including theoretical metrics
        output_dir: Directory to save visualization files
        
    Returns:
        List of paths to generated visualization files
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Set style
    plt.style.use('default')
    sns.set_palette("husl")
    
    generated_files = []
    
    # 1. Information-theoretic analysis
    info_path = output_dir / 'information_theoretic_analysis.png'
    _create_information_theoretic_plot(results, info_path)
    generated_files.append(str(info_path))
    
    # 2. Causal equivalence analysis
    causal_path = output_dir / 'causal_equivalence_analysis.png'
    _create_causal_equivalence_plot(results, causal_path)
    generated_files.append(str(causal_path))
    
    # 3. Optimality analysis
    optimality_path = output_dir / 'optimality_analysis.png'
    _create_optimality_plot(results, optimality_path)
    generated_files.append(str(optimality_path))
    
    # 4. Theoretical vs empirical comparison
    comparison_path = output_dir / 'theoretical_empirical_comparison.png'
    _create_theoretical_empirical_comparison(results, comparison_path)
    generated_files.append(str(comparison_path))
    
    plt.close('all')  # Clean up
    
    return generated_files


def _create_quality_gauge(ax, value: float, title: str):
    """Create a gauge-style plot for quality metrics."""
    # Create semicircle gauge
    theta = np.linspace(0, np.pi, 100)
    r_outer = 1.0
    r_inner = 0.7
    
    # Color gradient based on value
    if value >= 0.8:
        color = 'green'
    elif value >= 0.6:
        color = 'orange'
    elif value >= 0.4:
        color = 'yellow'
    else:
        color = 'red'
    
    # Draw gauge background
    ax.fill_between(theta, r_inner, r_outer, color='lightgray', alpha=0.3)
    
    # Draw filled portion based on value
    theta_fill = np.linspace(0, np.pi * value, int(100 * value))
    if len(theta_fill) > 0:
        ax.fill_between(theta_fill, r_inner, r_outer, color=color, alpha=0.7)
    
    # Draw needle
    needle_angle = np.pi * value
    needle_x = [0, np.cos(needle_angle) * 0.9]
    needle_y = [0, np.sin(needle_angle) * 0.9]
    ax.plot(needle_x, needle_y, 'k-', linewidth=3)
    ax.plot(0, 0, 'ko', markersize=8)
    
    # Labels
    ax.text(0, -0.3, f'{value:.3f}', ha='center', va='center', fontsize=14, fontweight='bold')
    ax.text(0, 1.2, title, ha='center', va='center', fontsize=12, fontweight='bold')
    
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-0.5, 1.3)
    ax.set_aspect('equal')
    ax.axis('off')


def _create_information_theoretic_plot(results: Dict[str, Any], output_path: Path):
    """Create information-theoretic measures visualization."""
    # Handle both direct and comprehensive analysis formats
    if 'theoretical_analysis' in results:
        info_results = results['theoretical_analysis'].get('information_theoretic_distance', {})
    else:
        info_results = results.get('information_theoretic_distance', {})
    
    if not info_results:
        return
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Information-Theoretic Analysis (ε-Machine Metrics)', fontsize=16, fontweight='bold')
    
    # 1. Statistical complexity comparison
    discovered_measures = info_results.get('discovered_measures', {})
    best_match = info_results.get('best_match', {})
    distances = info_results.get('distances', {})
    
    if discovered_measures and best_match:
        discovered_c = discovered_measures.get('statistical_complexity', 0.0)
        ground_truth_c = best_match.get('statistical_complexity', 0.0)
        complexity_ratio = info_results.get('theoretical_optimality_ratio', 0.0)
        
        categories = ['Discovered\nMachine', 'Ground Truth\nMachine']
        values = [discovered_c, ground_truth_c]
        colors = ['lightblue', 'lightcoral']
        
        bars = ax1.bar(categories, values, color=colors, alpha=0.7, edgecolor='black')
        ax1.set_title(f'Statistical Complexity (C_μ)\nRatio: {complexity_ratio:.3f}')
        ax1.set_ylabel('Complexity (bits)')
        
        # Add value labels
        for bar, value in zip(bars, values):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # 2. Entropy rate comparison
    if discovered_measures and best_match:
        discovered_h = discovered_measures.get('entropy_rate', 0.0)
        ground_truth_h = best_match.get('entropy_rate', 0.0)
        entropy_ratio = ground_truth_h / discovered_h if discovered_h > 0 else 0.0
        
        categories = ['Discovered\nMachine', 'Ground Truth\nMachine']
        values = [discovered_h, ground_truth_h]
        colors = ['lightgreen', 'orange']
        
        bars = ax2.bar(categories, values, color=colors, alpha=0.7, edgecolor='black')
        ax2.set_title(f'Entropy Rate (h_μ)\nRatio: {entropy_ratio:.3f}')
        ax2.set_ylabel('Entropy Rate (bits)')
        
        # Add value labels
        for bar, value in zip(bars, values):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # 3. Excess entropy comparison
    if discovered_measures and best_match:
        discovered_e = discovered_measures.get('excess_entropy', 0.0)
        ground_truth_e = best_match.get('excess_entropy', 0.0)
        excess_ratio = ground_truth_e / discovered_e if discovered_e > 0 else 0.0
        
        categories = ['Discovered\nMachine', 'Ground Truth\nMachine']
        values = [discovered_e, ground_truth_e]
        colors = ['purple', 'gold']
        
        bars = ax3.bar(categories, values, color=colors, alpha=0.7, edgecolor='black')
        ax3.set_title(f'Excess Entropy (E)\nRatio: {excess_ratio:.3f}')
        ax3.set_ylabel('Excess Entropy (bits)')
        
        # Add value labels
        for bar, value in zip(bars, values):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # 4. Information-theoretic summary
    ax4.axis('off')
    
    quality_assessment = info_results.get('quality_assessment', {})
    distance_score = info_results.get('combined_information_distance', 0.0)
    quality_score = quality_assessment.get('quality_score', 0.0)
    interpretation = quality_assessment.get('optimality_assessment', 'No interpretation available')
    
    # Get individual ratios for detailed summary
    c_ratio = complexity_ratio if discovered_measures and best_match else 0.0
    h_ratio = entropy_ratio if discovered_measures and best_match else 0.0
    e_ratio = excess_ratio if discovered_measures and best_match else 0.0
    
    summary_text = f"""Information-Theoretic Summary:
    
Distance Score: {distance_score:.4f}
Quality Score: {quality_score:.4f}

Ratio Analysis:
• Complexity Ratio: {c_ratio:.3f}
• Entropy Rate Ratio: {h_ratio:.3f}  
• Excess Entropy Ratio: {e_ratio:.3f}

Interpretation:
{interpretation}
"""
    
    ax4.text(0.05, 0.95, summary_text, transform=ax4.transAxes, fontsize=10,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def _create_causal_equivalence_plot(results: Dict[str, Any], output_path: Path):
    """Create causal equivalence analysis visualization."""
    # Handle both direct and comprehensive analysis formats
    if 'theoretical_analysis' in results:
        causal_results = results['theoretical_analysis'].get('causal_equivalence_distance', {})
    else:
        causal_results = results.get('causal_equivalence_distance', {})
    
    if not causal_results:
        return
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Causal Equivalence Analysis', fontsize=16, fontweight='bold')
    
    # 1. Equivalence strength gauge
    equivalence_strength = causal_results.get('causal_equivalence_score', 0.0)
    _create_quality_gauge(ax1, equivalence_strength, 'Causal Equivalence\nStrength')
    
    # 2. State refinement analysis
    refinement_data = causal_results.get('state_refinement_analysis', {})
    if refinement_data:
        correctly_identified = len(refinement_data.get('correctly_identified_states', []))
        over_split = len(refinement_data.get('over_split_states', []))
        under_split = len(refinement_data.get('under_split_states', []))
        
        discovered_states = correctly_identified + over_split + under_split
        minimal_states = correctly_identified  # Correctly identified represents optimal mapping
        refinement_ratio = minimal_states / discovered_states if discovered_states > 0 else 0.0
        
        categories = ['Discovered\nStates', 'Minimal\nStates']
        values = [discovered_states, minimal_states]
        colors = ['lightcoral', 'lightgreen']
        
        bars = ax2.bar(categories, values, color=colors, alpha=0.7, edgecolor='black')
        ax2.set_title(f'State Refinement Analysis\nRatio: {refinement_ratio:.3f}')
        ax2.set_ylabel('Number of States')
        
        # Add value labels
        for bar, value in zip(bars, values):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                    f'{int(value)}', ha='center', va='bottom', fontweight='bold')
    
    # 3. Future prediction consistency
    prediction_data = causal_results.get('future_prediction_consistency', {})
    if prediction_data and refinement_data:
        # Get consistency per correctly identified state
        correctly_identified_states = refinement_data.get('correctly_identified_states', [])
        state_names = [state['discovered_state'] for state in correctly_identified_states]
        consistency_scores = [state['equivalence_strength'] for state in correctly_identified_states]
        
        if consistency_scores:
            bars = ax3.bar(range(len(state_names)), consistency_scores, 
                          color='skyblue', alpha=0.7, edgecolor='black')
            ax3.set_title('Future Prediction Consistency\nper Discovered State')
            ax3.set_xlabel('Discovered States')
            ax3.set_ylabel('Consistency Score')
            ax3.set_xticks(range(len(state_names)))
            ax3.set_xticklabels(state_names, rotation=45)
            
            # Add value labels
            for bar, score in zip(bars, consistency_scores):
                ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                        f'{score:.3f}', ha='center', va='bottom', fontsize=8)
            
            # Add average line
            avg_consistency = np.mean(consistency_scores)
            ax3.axhline(avg_consistency, color='red', linestyle='--', 
                       label=f'Average: {avg_consistency:.3f}')
            ax3.legend()
    
    # 4. Causal equivalence summary
    ax4.axis('off')
    
    quality_assessment = causal_results.get('quality_assessment', {})
    distance_score = 1.0 - equivalence_strength  # Distance is inverse of equivalence strength
    quality_score = quality_assessment.get('quality_score', equivalence_strength)
    interpretation = quality_assessment.get('description', 'Good causal equivalence recovery')
    
    # Additional metrics
    avg_consistency = prediction_data.get('consistency_score', 0.0) if prediction_data else 0.0
    if consistency_scores:
        avg_equivalence = np.mean(consistency_scores)
    else:
        avg_equivalence = 0.0
    
    summary_text = f"""Causal Equivalence Summary:
    
Distance Score: {distance_score:.4f}
Quality Score: {quality_score:.4f}
Equivalence Strength: {equivalence_strength:.4f}

Analysis Details:
• Avg. Prediction Consistency: {avg_consistency:.3f}
• Avg. Equivalence Strength: {avg_equivalence:.3f}
• State Refinement Ratio: {refinement_ratio:.3f}

Interpretation:
{interpretation}
"""
    
    ax4.text(0.05, 0.95, summary_text, transform=ax4.transAxes, fontsize=10,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.3))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def _create_optimality_plot(results: Dict[str, Any], output_path: Path):
    """Create optimality analysis visualization."""
    # Handle both direct and comprehensive analysis formats
    if 'theoretical_analysis' in results:
        optimality_results = results['theoretical_analysis'].get('optimality_analysis', {})
    else:
        optimality_results = results.get('optimality_analysis', {})
    
    if not optimality_results:
        return
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('ε-Machine Optimality Analysis', fontsize=16, fontweight='bold')
    
    # 1. Optimality scores radar chart
    if optimality_results:
        categories = ['Minimality', 'Unifilarity', 'Causal\nSufficiency']
        values = [
            optimality_results.get('minimality_score', 0.0),
            optimality_results.get('unifilarity_score', 0.0),
            optimality_results.get('causal_sufficiency_score', 0.0)
        ]
        
        # Create radar chart
        angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
        values += values[:1]  # Complete the circle
        angles += angles[:1]
        
        ax1.plot(angles, values, 'o-', linewidth=2, color='blue', alpha=0.7)
        ax1.fill(angles, values, alpha=0.25, color='blue')
        ax1.set_xticks(angles[:-1])
        ax1.set_xticklabels(categories)
        ax1.set_ylim(0, 1)
        ax1.set_title('Optimality Properties\nRadar Chart')
        ax1.grid(True)
    
    # 2. Theoretical bounds comparison
    bounds_data = optimality_results.get('theoretical_bounds', {})
    if bounds_data:
        # Get discovered complexity from the analysis (statistical complexity)
        discovered_info = optimality_results.get('discovered_machine_info', {})
        discovered_complexity = len(discovered_info.get('states', {}))  # Number of states as complexity proxy
        
        theoretical_minimum = bounds_data.get('minimum_possible_states', 0)
        optimal_complexity = bounds_data.get('optimal_statistical_complexity', 0.0)
        efficiency_ratio = theoretical_minimum / discovered_complexity if discovered_complexity > 0 else 0.0
        
        categories = ['Theoretical\nMinimum', 'Discovered\nComplexity']
        values = [theoretical_minimum, discovered_complexity]
        colors = ['lightgreen', 'lightcoral']
        
        bars = ax2.bar(categories, values, color=colors, alpha=0.7, edgecolor='black')
        ax2.set_title(f'Complexity vs Theoretical Bounds\nEfficiency: {efficiency_ratio:.3f}')
        ax2.set_ylabel('Statistical Complexity (bits)')
        
        # Add value labels
        for bar, value in zip(bars, values):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # Add efficiency annotation
        ax2.axhline(theoretical_minimum, color='green', linestyle='--', alpha=0.7, 
                   label='Theoretical Optimum')
        ax2.legend()
    
    # 3. Property breakdown
    if optimality_results:
        properties = ['Minimality', 'Unifilarity', 'Causal\nSufficiency']
        scores = [
            optimality_results.get('minimality_score', 0.0),
            optimality_results.get('unifilarity_score', 0.0),  
            optimality_results.get('causal_sufficiency_score', 0.0)
        ]
        
        bars = ax3.barh(properties, scores, color='lightblue', alpha=0.7, edgecolor='black')
        ax3.set_title('Optimality Property Breakdown')
        ax3.set_xlabel('Property Score')
        ax3.set_xlim(0, 1)
        
        # Add value labels
        for bar, score in zip(bars, scores):
            ax3.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height()/2,
                    f'{score:.3f}', ha='left', va='center', fontweight='bold')
    
    # 4. Optimality summary
    ax4.axis('off')
    
    # Calculate overall optimality score
    overall_optimality = np.mean([
        optimality_results.get('minimality_score', 0.0),
        optimality_results.get('unifilarity_score', 0.0),
        optimality_results.get('causal_sufficiency_score', 0.0)
    ])
    
    distance_score = 1.0 - overall_optimality
    quality_score = overall_optimality
    prediction_opt = optimality_results.get('prediction_optimality', 0.0)
    
    # Interpretation based on scores
    if overall_optimality >= 0.8:
        interpretation = "Excellent optimality - near-optimal ε-machine"
    elif overall_optimality >= 0.6:
        interpretation = "Good optimality - well-structured ε-machine"
    else:
        interpretation = "Fair optimality - suboptimal structure detected"
    
    summary_text = f"""Optimality Analysis Summary:
    
Distance Score: {distance_score:.4f}
Quality Score: {quality_score:.4f}
Overall Optimality: {overall_optimality:.4f}

Efficiency Analysis:
• Theoretical Efficiency: {efficiency_ratio:.3f}

Property Scores:
• Minimality: {optimality_results.get('minimality_score', 0.0):.3f}
• Unifilarity: {optimality_results.get('unifilarity_score', 0.0):.3f}
• Causal Sufficiency: {optimality_results.get('causal_sufficiency_score', 0.0):.3f}
• Prediction Optimality: {prediction_opt:.3f}

Interpretation:
{interpretation}
"""
    
    ax4.text(0.05, 0.95, summary_text, transform=ax4.transAxes, fontsize=9,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.3))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def _create_theoretical_empirical_comparison(results: Dict[str, Any], output_path: Path):
    """Create comparison between theoretical and empirical approaches."""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Theoretical vs Empirical Approach Comparison', fontsize=16, fontweight='bold')
    
    # 1. Quality score comparison
    summary = results.get('summary', {})
    empirical_quality = 0.0
    theoretical_quality = 0.0
    
    # Extract empirical quality (average of empirical metrics)
    if 'empirical_analysis' in results:
        empirical_data = results['empirical_analysis']
    else:
        empirical_data = results
        
    empirical_metrics = ['symbol_distribution_distance', 'state_mapping_distance', 'transition_structure_distance']
    empirical_scores = []
    for metric in empirical_metrics:
        metric_data = empirical_data.get(metric, {})
        if 'quality_assessment' in metric_data:
            empirical_scores.append(metric_data['quality_assessment'].get('overall_quality_score', 0.0))
        elif 'assignment_quality' in metric_data:
            empirical_scores.append(metric_data['assignment_quality'].get('quality_score', 0.0))
    
    if empirical_scores:
        empirical_quality = np.mean(empirical_scores)
    
    # Extract theoretical quality (average of theoretical metrics)
    if 'theoretical_analysis' in results:
        theoretical_data = results['theoretical_analysis']
    else:
        theoretical_data = results
        
    theoretical_metrics = ['information_theoretic_distance', 'causal_equivalence_distance', 'optimality_analysis']
    theoretical_scores = []
    for metric in theoretical_metrics:
        metric_data = theoretical_data.get(metric, {})
        if metric == 'information_theoretic_distance':
            quality_assessment = metric_data.get('quality_assessment', {})
            theoretical_scores.append(quality_assessment.get('quality_score', 0.0))
        elif metric == 'causal_equivalence_distance':
            quality_assessment = metric_data.get('quality_assessment', {})
            equiv_score = metric_data.get('causal_equivalence_score', 0.0)
            theoretical_scores.append(quality_assessment.get('quality_score', equiv_score))
        elif metric == 'optimality_analysis':
            # Calculate overall optimality from individual scores
            overall_optimality = np.mean([
                metric_data.get('minimality_score', 0.0),
                metric_data.get('unifilarity_score', 0.0),
                metric_data.get('causal_sufficiency_score', 0.0)
            ])
            theoretical_scores.append(overall_optimality)
    
    if theoretical_scores:
        theoretical_quality = np.mean(theoretical_scores)
    
    categories = ['Empirical\nApproach', 'Theoretical\nApproach']
    values = [empirical_quality, theoretical_quality]
    colors = ['lightblue', 'lightcoral']
    
    bars = ax1.bar(categories, values, color=colors, alpha=0.7, edgecolor='black')
    ax1.set_title('Quality Score Comparison')
    ax1.set_ylabel('Average Quality Score')
    ax1.set_ylim(0, 1)
    
    # Add value labels
    for bar, value in zip(bars, values):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # 2. Agreement analysis
    agreement_score = abs(empirical_quality - theoretical_quality)
    agreement_level = 1.0 - agreement_score  # Higher when approaches agree
    
    _create_quality_gauge(ax2, agreement_level, 'Methodological\nAgreement')
    
    # 3. Metric breakdown comparison
    all_metrics = ['Symbol\nDistribution', 'State\nMapping', 'Transition\nStructure', 
                   'Information\nTheoretic', 'Causal\nEquivalence', 'Optimality']
    all_scores = empirical_scores + theoretical_scores
    
    if len(all_scores) == len(all_metrics):
        colors_breakdown = ['lightblue'] * 3 + ['lightcoral'] * 3
        bars = ax3.bar(range(len(all_metrics)), all_scores, color=colors_breakdown, 
                      alpha=0.7, edgecolor='black')
        ax3.set_title('Individual Metric Comparison')
        ax3.set_ylabel('Quality Score')
        ax3.set_xticks(range(len(all_metrics)))
        ax3.set_xticklabels(all_metrics, rotation=45, ha='right')
        ax3.set_ylim(0, 1)
        
        # Add dividing line
        ax3.axvline(2.5, color='black', linestyle='--', alpha=0.5)
        ax3.text(1, 0.9, 'Empirical', ha='center', fontweight='bold', 
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
        ax3.text(4, 0.9, 'Theoretical', ha='center', fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.5))
    
    # 4. Summary and interpretation
    ax4.axis('off')
    
    # Get summary data from comprehensive analysis or create from computed values
    if 'combined_assessment' in results:
        combined = results['combined_assessment']
        overall_quality = combined.get('consensus_quality_score', np.mean([empirical_quality, theoretical_quality]))
        confidence = combined.get('confidence_level', 0.0)
        interpretation_data = combined.get('research_interpretation', {})
    else:
        overall_quality = summary.get('overall_quality_score', np.mean([empirical_quality, theoretical_quality]))
        confidence = summary.get('confidence', 0.0)
        interpretation_data = summary.get('interpretation', {})
    
    # Determine agreement level description
    if agreement_level >= 0.8:
        agreement_desc = "High Agreement"
    elif agreement_level >= 0.6:
        agreement_desc = "Moderate Agreement"
    elif agreement_level >= 0.4:
        agreement_desc = "Low Agreement"
    else:
        agreement_desc = "Poor Agreement"
    
    summary_text = f"""Comparative Analysis Summary:
    
Overall Quality: {overall_quality:.4f}
Confidence: {confidence:.4f}
Methodological Agreement: {agreement_level:.4f} ({agreement_desc})

Approach Scores:
• Empirical Quality: {empirical_quality:.3f}
• Theoretical Quality: {theoretical_quality:.3f}
• Score Difference: {agreement_score:.3f}

Interpretation:
{interpretation_data.get('overall_assessment', 'Comprehensive analysis shows good methodological agreement') if isinstance(interpretation_data, dict) else str(interpretation_data)}

Confidence Assessment:
{interpretation_data.get('confidence_assessment', 'High confidence in combined results') if isinstance(interpretation_data, dict) else 'High confidence in combined results'}
"""
    
    ax4.text(0.05, 0.95, summary_text, transform=ax4.transAxes, fontsize=9,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.3))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()