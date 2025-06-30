"""Visualization utilities for machine distance analysis."""

import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Any, Optional
import seaborn as sns
from pathlib import Path


def create_distance_visualizations(results: Dict[str, Any], output_dir: str) -> List[str]:
    """
    Create comprehensive visualizations for machine distance analysis.
    
    Args:
        results: Results from MachineDistanceCalculator.compute_all_distances
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
    
    # 1. Summary metrics overview
    summary_path = output_dir / 'summary_metrics.png'
    _create_summary_metrics_plot(results, summary_path)
    generated_files.append(str(summary_path))
    
    # 2. Symbol distribution analysis
    symbol_path = output_dir / 'symbol_distribution_analysis.png'
    _create_symbol_distribution_plot(results, symbol_path)
    generated_files.append(str(symbol_path))
    
    # 3. State mapping visualization
    mapping_path = output_dir / 'state_mapping_analysis.png'
    _create_state_mapping_plot(results, mapping_path)
    generated_files.append(str(mapping_path))
    
    # 4. Detailed state comparison heatmap
    heatmap_path = output_dir / 'state_comparison_heatmap.png'
    _create_state_comparison_heatmap(results, heatmap_path)
    generated_files.append(str(heatmap_path))
    
    plt.close('all')  # Clean up
    
    return generated_files


def _create_summary_metrics_plot(results: Dict[str, Any], output_path: Path):
    """Create overview plot of all distance metrics."""
    summary = results.get('summary', {})
    metric_scores = summary.get('metric_scores', {})
    
    if not metric_scores:
        return
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Machine Distance Analysis Summary', fontsize=16, fontweight='bold')
    
    # 1. Metric scores bar chart
    metrics = list(metric_scores.keys())
    scores = list(metric_scores.values())
    
    bars = ax1.bar(metrics, scores, color=['skyblue', 'lightcoral', 'lightgreen'])
    ax1.set_title('Distance Metric Quality Scores')
    ax1.set_ylabel('Quality Score')
    ax1.set_ylim(0, 1)
    ax1.tick_params(axis='x', rotation=45)
    
    # Add value labels on bars
    for bar, score in zip(bars, scores):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{score:.3f}', ha='center', va='bottom')
    
    # 2. Overall quality gauge
    overall_quality = summary.get('overall_quality_score', 0.0)
    _create_quality_gauge(ax2, overall_quality, 'Overall Quality')
    
    # 3. Confidence gauge
    confidence = summary.get('confidence', 0.0)
    _create_quality_gauge(ax3, confidence, 'Confidence Level')
    
    # 4. Interpretation text
    ax4.axis('off')
    interpretation = results.get('summary', {}).get('interpretation', {})
    quality_text = interpretation.get('overall_quality', 'No interpretation available')
    confidence_text = interpretation.get('confidence_level', '')
    
    ax4.text(0.05, 0.9, 'Interpretation:', fontweight='bold', fontsize=12, transform=ax4.transAxes)
    ax4.text(0.05, 0.7, quality_text, fontsize=10, wrap=True, transform=ax4.transAxes)
    ax4.text(0.05, 0.4, confidence_text, fontsize=10, wrap=True, transform=ax4.transAxes)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


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


def _create_symbol_distribution_plot(results: Dict[str, Any], output_path: Path):
    """Create detailed symbol distribution analysis plot."""
    symbol_results = results.get('symbol_distribution_distance', {})
    
    if not symbol_results:
        return
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Symbol Distribution Analysis', fontsize=16, fontweight='bold')
    
    # 1. JS divergence distribution
    mappings = symbol_results.get('state_mappings', [])
    if mappings:
        js_divergences = [m.get('js_divergence', 0.0) for m in mappings]
        
        ax1.hist(js_divergences, bins=10, alpha=0.7, color='skyblue', edgecolor='black')
        ax1.axvline(np.mean(js_divergences), color='red', linestyle='--', 
                   label=f'Mean: {np.mean(js_divergences):.4f}')
        ax1.set_title('JS Divergence Distribution')
        ax1.set_xlabel('Jensen-Shannon Divergence')
        ax1.set_ylabel('Count')
        ax1.legend()
    
    # 2. Best matches bar chart
    if mappings:
        state_names = [m.get('discovered_state', f'State_{i}') for i, m in enumerate(mappings)]
        divergences = [m.get('js_divergence', 0.0) for m in mappings]
        
        bars = ax2.bar(range(len(state_names)), divergences, color='lightcoral')
        ax2.set_title('State-wise JS Divergences')
        ax2.set_xlabel('Discovered States')
        ax2.set_ylabel('JS Divergence')
        ax2.set_xticks(range(len(state_names)))
        ax2.set_xticklabels(state_names, rotation=45)
        
        # Add value labels
        for i, (bar, div) in enumerate(zip(bars, divergences)):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                    f'{div:.3f}', ha='center', va='bottom', fontsize=8)
    
    # 3. Coverage analysis
    coverage = symbol_results.get('coverage_score', {})
    total_gt = coverage.get('total_ground_truth_states', 0)
    matched = coverage.get('well_matched_states', 0)
    unmatched = total_gt - matched
    
    if total_gt > 0:
        labels = ['Well Matched', 'Poorly Matched']
        sizes = [matched, unmatched]
        colors = ['lightgreen', 'lightcoral']
        
        ax3.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
        ax3.set_title(f'Ground Truth State Coverage\n({matched}/{total_gt} states well matched)')
    
    # 4. Quality assessment
    quality = symbol_results.get('quality_assessment', {})
    metrics = ['Overall Quality', 'Divergence Quality', 'Coverage Quality', 'Consistency']
    values = [
        quality.get('overall_quality_score', 0.0),
        quality.get('divergence_quality', 0.0),
        quality.get('coverage_quality', 0.0),
        quality.get('consistency_score', 0.0)
    ]
    
    bars = ax4.barh(metrics, values, color='lightblue')
    ax4.set_title('Quality Assessment Breakdown')
    ax4.set_xlabel('Score')
    ax4.set_xlim(0, 1)
    
    # Add value labels
    for bar, value in zip(bars, values):
        ax4.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                f'{value:.3f}', ha='left', va='center')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def _create_state_mapping_plot(results: Dict[str, Any], output_path: Path):
    """Create state mapping analysis visualization."""
    mapping_results = results.get('state_mapping_distance', {})
    
    if not mapping_results:
        return
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('State Mapping Analysis', fontsize=16, fontweight='bold')
    
    # 1. Assignment costs
    assignments = mapping_results.get('optimal_assignment', [])
    if assignments:
        costs = [a.get('cost', 0.0) for a in assignments]
        state_names = [a.get('discovered_state', f'State_{i}') for i, a in enumerate(assignments)]
        
        bars = ax1.bar(range(len(state_names)), costs, color='lightcoral')
        ax1.set_title('Assignment Costs (Lower is Better)')
        ax1.set_xlabel('Discovered States')
        ax1.set_ylabel('Assignment Cost')
        ax1.set_xticks(range(len(state_names)))
        ax1.set_xticklabels(state_names, rotation=45)
        
        # Add value labels
        for bar, cost in zip(bars, costs):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                    f'{cost:.3f}', ha='center', va='bottom', fontsize=8)
    
    # 2. Assignment network diagram
    if assignments:
        _create_assignment_network(ax2, assignments)
    
    # 3. Cost distribution
    if assignments:
        costs = [a.get('cost', 0.0) for a in assignments]
        ax3.hist(costs, bins=max(5, len(costs)//2), alpha=0.7, color='skyblue', edgecolor='black')
        ax3.axvline(np.mean(costs), color='red', linestyle='--', 
                   label=f'Mean: {np.mean(costs):.4f}')
        ax3.set_title('Assignment Cost Distribution')
        ax3.set_xlabel('Assignment Cost')
        ax3.set_ylabel('Count')
        ax3.legend()
    
    # 4. Summary statistics
    ax4.axis('off')
    total_cost = mapping_results.get('total_cost', 0.0)
    avg_cost = mapping_results.get('average_cost', 0.0)
    quality = mapping_results.get('assignment_quality', {})
    unmatched_disc = len(mapping_results.get('unmatched_discovered_states', []))
    unmatched_true = len(mapping_results.get('unmatched_true_states', []))
    
    stats_text = f"""Assignment Summary:
    
Total Cost: {total_cost:.4f}
Average Cost: {avg_cost:.4f}
Quality Score: {quality.get('quality_score', 0.0):.3f}
Confidence: {quality.get('confidence', 0.0):.3f}

Unmatched States:
- Discovered: {unmatched_disc}
- Ground Truth: {unmatched_true}
"""
    
    ax4.text(0.05, 0.95, stats_text, transform=ax4.transAxes, fontsize=11,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def _create_assignment_network(ax, assignments):
    """Create a network diagram showing state assignments."""
    # Simple network visualization
    discovered_states = list(set(a.get('discovered_state', 'Unknown') for a in assignments))
    ground_truth_states = list(set(a.get('ground_truth_state', 'Unknown') for a in assignments))
    
    # Position nodes
    disc_y = np.linspace(0.8, 0.2, len(discovered_states))
    gt_y = np.linspace(0.8, 0.2, len(ground_truth_states))
    
    # Draw discovered states (left side)
    for i, state in enumerate(discovered_states):
        ax.scatter(0.2, disc_y[i], s=100, c='lightblue', edgecolors='black')
        ax.text(0.15, disc_y[i], state, ha='right', va='center', fontsize=8)
    
    # Draw ground truth states (right side)  
    for i, state in enumerate(ground_truth_states):
        ax.scatter(0.8, gt_y[i], s=100, c='lightcoral', edgecolors='black')
        ax.text(0.85, gt_y[i], state, ha='left', va='center', fontsize=8)
    
    # Draw assignment lines
    for assignment in assignments:
        disc_state = assignment.get('discovered_state', 'Unknown')
        gt_state = assignment.get('ground_truth_state', 'Unknown')
        cost = assignment.get('cost', 0.0)
        
        if disc_state in discovered_states and gt_state in ground_truth_states:
            disc_idx = discovered_states.index(disc_state)
            gt_idx = ground_truth_states.index(gt_state)
            
            # Line thickness inversely proportional to cost
            linewidth = max(0.5, 3 * (1 - cost))
            color = 'green' if cost < 0.3 else 'orange' if cost < 0.6 else 'red'
            
            ax.plot([0.2, 0.8], [disc_y[disc_idx], gt_y[gt_idx]], 
                   color=color, linewidth=linewidth, alpha=0.7)
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title('State Assignment Network')
    ax.text(0.2, 0.95, 'Discovered', ha='center', va='center', fontweight='bold')
    ax.text(0.8, 0.95, 'Ground Truth', ha='center', va='center', fontweight='bold')
    ax.axis('off')


def _create_state_comparison_heatmap(results: Dict[str, Any], output_path: Path):
    """Create detailed heatmap comparing state distributions."""
    symbol_results = results.get('symbol_distribution_distance', {})
    mappings = symbol_results.get('state_mappings', [])
    
    if not mappings:
        return
    
    fig, axes = plt.subplots(1, len(mappings), figsize=(4*len(mappings), 6))
    if len(mappings) == 1:
        axes = [axes]
    
    fig.suptitle('State-wise Symbol Distribution Comparison', fontsize=16, fontweight='bold')
    
    for i, (ax, mapping) in enumerate(zip(axes, mappings)):
        disc_dist = mapping.get('discovered_distribution', {})
        gt_dist = mapping.get('matched_distribution', {})
        
        if not disc_dist or not gt_dist:
            continue
        
        symbols = sorted(set(disc_dist.keys()) | set(gt_dist.keys()))
        disc_probs = [disc_dist.get(symbol, 0.0) for symbol in symbols]
        gt_probs = [gt_dist.get(symbol, 0.0) for symbol in symbols]
        
        # Create comparison matrix
        data = np.array([disc_probs, gt_probs])
        
        im = ax.imshow(data, cmap='Blues', aspect='auto', vmin=0, vmax=1)
        
        # Add text annotations
        for row in range(2):
            for col in range(len(symbols)):
                text = ax.text(col, row, f'{data[row, col]:.3f}',
                             ha="center", va="center", color="black", fontweight='bold')
        
        ax.set_title(f"{mapping.get('discovered_state', 'Unknown')}\n→ {mapping.get('best_match', {}).get('full_name', 'Unknown')}\nJS: {mapping.get('js_divergence', 0.0):.4f}")
        ax.set_xticks(range(len(symbols)))
        ax.set_xticklabels(symbols)
        ax.set_yticks([0, 1])
        ax.set_yticklabels(['Discovered', 'Ground Truth'])
        
        # Add colorbar for first subplot
        if i == 0:
            cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label('Probability', rotation=270, labelpad=15)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()