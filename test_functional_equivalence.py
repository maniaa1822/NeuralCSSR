#!/usr/bin/env python3
"""
Test functional equivalence between ground truth and extracted FSMs.

Two FSMs are functionally equivalent if they generate the same sequence
distributions, even if their internal structure differs. This is a much
more meaningful test than exact probability matching.
"""

import json
import numpy as np
import random
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict, Counter
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats


class FSMSimulator:
    """Simulate sequence generation from FSM structure."""
    
    def __init__(self, fsm_data: Dict, fsm_type: str = "ground_truth"):
        self.fsm_data = fsm_data
        self.fsm_type = fsm_type
        
        if fsm_type == "ground_truth":
            self._setup_ground_truth()
        else:
            self._setup_extracted()
    
    def _setup_ground_truth(self):
        """Setup ground truth FSM for simulation."""
        self.states = self.fsm_data['states']
        self.start_state = self.fsm_data['start_state']
        self.transitions = {}
        
        # Parse transition format: "state|token" -> [{"to_state": ..., "probability": ...}]
        for trans_key, trans_list in self.fsm_data['transitions'].items():
            if '|' in trans_key:
                from_state, token = trans_key.split('|')
                if from_state not in self.transitions:
                    self.transitions[from_state] = {}
                
                self.transitions[from_state][token] = []
                for trans in trans_list:
                    self.transitions[from_state][token].append({
                        'to_state': trans['to_state'],
                        'probability': trans['probability']
                    })
    
    def _setup_extracted(self):
        """Setup extracted FSM for simulation."""
        fsm_structure = self.fsm_data['epsilon_machine']
        self.states = fsm_structure['states']
        self.start_state = self.states[0]  # Assume first state is start state
        self.transitions = {}
        
        # Convert extracted format to simulation format
        for state_name, state_transitions in fsm_structure['transitions'].items():
            self.transitions[state_name] = {}
            
            for token in ['0', '1']:
                if token in state_transitions:
                    self.transitions[state_name][token] = []
                    for target_state, prob in state_transitions[token].items():
                        if prob > 0:  # Only include non-zero transitions
                            self.transitions[state_name][token].append({
                                'to_state': target_state,
                                'probability': prob
                            })
    
    def generate_sequence(self, length: int, seed: int = None) -> List[str]:
        """Generate a sequence from the FSM."""
        if seed is not None:
            random.seed(seed)
        
        sequence = []
        current_state = self.start_state
        
        for _ in range(length):
            # Choose next token based on current state
            available_tokens = []
            if current_state in self.transitions:
                available_tokens = list(self.transitions[current_state].keys())
            
            if not available_tokens:
                # If no transitions available, choose randomly
                next_token = random.choice(['0', '1'])
            else:
                # Weight by probability of transitions
                token_weights = []
                for token in available_tokens:
                    total_prob = sum(t['probability'] for t in self.transitions[current_state][token])
                    token_weights.append(total_prob)
                
                if sum(token_weights) > 0:
                    next_token = random.choices(available_tokens, weights=token_weights)[0]
                else:
                    next_token = random.choice(available_tokens)
            
            sequence.append(next_token)
            
            # Transition to next state
            if current_state in self.transitions and next_token in self.transitions[current_state]:
                transitions = self.transitions[current_state][next_token]
                if transitions:
                    probs = [t['probability'] for t in transitions]
                    states = [t['to_state'] for t in transitions]
                    if sum(probs) > 0:
                        current_state = random.choices(states, weights=probs)[0]
                    else:
                        current_state = random.choice(states)
        
        return sequence


def compute_sequence_statistics(sequences: List[List[str]]) -> Dict:
    """Compute comprehensive statistics from generated sequences."""
    
    # Flatten all sequences
    all_symbols = [symbol for seq in sequences for symbol in seq]
    
    # Basic symbol distribution
    symbol_counts = Counter(all_symbols)
    total_symbols = len(all_symbols)
    symbol_dist = {sym: count/total_symbols for sym, count in symbol_counts.items()}
    
    # N-gram distributions (1-gram to 5-gram)
    ngram_stats = {}
    for n in range(1, 6):
        ngram_counts = defaultdict(int)
        total_ngrams = 0
        
        for seq in sequences:
            for i in range(len(seq) - n + 1):
                ngram = ''.join(seq[i:i+n])
                ngram_counts[ngram] += 1
                total_ngrams += 1
        
        ngram_dist = {ngram: count/total_ngrams for ngram, count in ngram_counts.items()}
        ngram_stats[f'{n}gram'] = {
            'distribution': ngram_dist,
            'unique_patterns': len(ngram_counts),
            'total_count': total_ngrams
        }
    
    # Transition probabilities
    transition_counts = defaultdict(lambda: defaultdict(int))
    total_transitions = 0
    
    for seq in sequences:
        for i in range(len(seq) - 1):
            current = seq[i]
            next_sym = seq[i + 1]
            transition_counts[current][next_sym] += 1
            total_transitions += 1
    
    transition_probs = {}
    for current in transition_counts:
        total_from_current = sum(transition_counts[current].values())
        transition_probs[current] = {
            next_sym: count / total_from_current 
            for next_sym, count in transition_counts[current].items()
        }
    
    # Run length statistics
    run_lengths = {'0': [], '1': []}
    for seq in sequences:
        current_symbol = seq[0] if seq else '0'
        current_run = 1
        
        for i in range(1, len(seq)):
            if seq[i] == current_symbol:
                current_run += 1
            else:
                run_lengths[current_symbol].append(current_run)
                current_symbol = seq[i]
                current_run = 1
        
        if seq:
            run_lengths[current_symbol].append(current_run)
    
    return {
        'symbol_distribution': symbol_dist,
        'ngram_statistics': ngram_stats,
        'transition_probabilities': transition_probs,
        'run_length_stats': {
            '0': {
                'mean': np.mean(run_lengths['0']) if run_lengths['0'] else 0,
                'std': np.std(run_lengths['0']) if run_lengths['0'] else 0,
                'max': max(run_lengths['0']) if run_lengths['0'] else 0
            },
            '1': {
                'mean': np.mean(run_lengths['1']) if run_lengths['1'] else 0,
                'std': np.std(run_lengths['1']) if run_lengths['1'] else 0,
                'max': max(run_lengths['1']) if run_lengths['1'] else 0
            }
        },
        'total_symbols': total_symbols,
        'num_sequences': len(sequences)
    }


def compare_distributions(stats1: Dict, stats2: Dict, name1: str, name2: str) -> Dict:
    """Compare statistical distributions between two FSMs."""
    
    comparisons = {}
    
    # Symbol distribution comparison
    symbols = set(stats1['symbol_distribution'].keys()) | set(stats2['symbol_distribution'].keys())
    symbol_diffs = {}
    for sym in symbols:
        prob1 = stats1['symbol_distribution'].get(sym, 0)
        prob2 = stats2['symbol_distribution'].get(sym, 0)
        symbol_diffs[sym] = abs(prob1 - prob2)
    
    comparisons['symbol_distribution'] = {
        'differences': symbol_diffs,
        'max_diff': max(symbol_diffs.values()) if symbol_diffs else 0,
        'mean_diff': np.mean(list(symbol_diffs.values())) if symbol_diffs else 0
    }
    
    # N-gram distribution comparisons
    ngram_comparisons = {}
    for n in range(1, 6):
        ngram_key = f'{n}gram'
        if ngram_key in stats1['ngram_statistics'] and ngram_key in stats2['ngram_statistics']:
            dist1 = stats1['ngram_statistics'][ngram_key]['distribution']
            dist2 = stats2['ngram_statistics'][ngram_key]['distribution']
            
            all_ngrams = set(dist1.keys()) | set(dist2.keys())
            ngram_diffs = {}
            for ngram in all_ngrams:
                prob1 = dist1.get(ngram, 0)
                prob2 = dist2.get(ngram, 0)
                ngram_diffs[ngram] = abs(prob1 - prob2)
            
            # KL divergence
            kl_div = 0
            for ngram in all_ngrams:
                p1 = dist1.get(ngram, 1e-10)
                p2 = dist2.get(ngram, 1e-10)
                if p1 > 0:
                    kl_div += p1 * np.log(p1 / p2)
            
            ngram_comparisons[ngram_key] = {
                'max_diff': max(ngram_diffs.values()) if ngram_diffs else 0,
                'mean_diff': np.mean(list(ngram_diffs.values())) if ngram_diffs else 0,
                'kl_divergence': kl_div,
                'unique_patterns_1': stats1['ngram_statistics'][ngram_key]['unique_patterns'],
                'unique_patterns_2': stats2['ngram_statistics'][ngram_key]['unique_patterns']
            }
    
    comparisons['ngram_comparisons'] = ngram_comparisons
    
    # Transition probability comparison
    trans_diffs = {}
    all_symbols = set(stats1['transition_probabilities'].keys()) | set(stats2['transition_probabilities'].keys())
    
    for sym in all_symbols:
        trans1 = stats1['transition_probabilities'].get(sym, {})
        trans2 = stats2['transition_probabilities'].get(sym, {})
        
        all_next = set(trans1.keys()) | set(trans2.keys())
        for next_sym in all_next:
            prob1 = trans1.get(next_sym, 0)
            prob2 = trans2.get(next_sym, 0)
            key = f"{sym}->{next_sym}"
            trans_diffs[key] = abs(prob1 - prob2)
    
    comparisons['transition_probabilities'] = {
        'differences': trans_diffs,
        'max_diff': max(trans_diffs.values()) if trans_diffs else 0,
        'mean_diff': np.mean(list(trans_diffs.values())) if trans_diffs else 0
    }
    
    return comparisons


def create_comparison_visualizations(gt_stats: Dict, ext_stats: Dict, comparisons: Dict, 
                                   output_dir: Path) -> None:
    """Create visualizations comparing the statistical properties."""
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Symbol distribution comparison
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Symbol distributions
    symbols = ['0', '1']
    gt_probs = [gt_stats['symbol_distribution'].get(s, 0) for s in symbols]
    ext_probs = [ext_stats['symbol_distribution'].get(s, 0) for s in symbols]
    
    x = np.arange(len(symbols))
    width = 0.35
    
    axes[0, 0].bar(x - width/2, gt_probs, width, label='Ground Truth', alpha=0.8)
    axes[0, 0].bar(x + width/2, ext_probs, width, label='Extracted', alpha=0.8)
    axes[0, 0].set_title('Symbol Distribution Comparison')
    axes[0, 0].set_xlabel('Symbol')
    axes[0, 0].set_ylabel('Probability')
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels(symbols)
    axes[0, 0].legend()
    
    # Transition probabilities
    transitions = ['0->0', '0->1', '1->0', '1->1']
    gt_trans = []
    ext_trans = []
    
    for trans in transitions:
        from_sym, to_sym = trans.split('->')
        gt_prob = gt_stats['transition_probabilities'].get(from_sym, {}).get(to_sym, 0)
        ext_prob = ext_stats['transition_probabilities'].get(from_sym, {}).get(to_sym, 0)
        gt_trans.append(gt_prob)
        ext_trans.append(ext_prob)
    
    x = np.arange(len(transitions))
    axes[0, 1].bar(x - width/2, gt_trans, width, label='Ground Truth', alpha=0.8)
    axes[0, 1].bar(x + width/2, ext_trans, width, label='Extracted', alpha=0.8)
    axes[0, 1].set_title('Transition Probability Comparison')
    axes[0, 1].set_xlabel('Transition')
    axes[0, 1].set_ylabel('Probability')
    axes[0, 1].set_xticks(x)
    axes[0, 1].set_xticklabels(transitions, rotation=45)
    axes[0, 1].legend()
    
    # N-gram comparison (3-grams)
    if '3gram' in comparisons['ngram_comparisons']:
        ngram_comp = comparisons['ngram_comparisons']['3gram']
        axes[1, 0].bar(['Max Diff', 'Mean Diff', 'KL Div'], 
                      [ngram_comp['max_diff'], ngram_comp['mean_diff'], 
                       min(ngram_comp['kl_divergence'], 1.0)],  # Cap KL div for visualization
                      alpha=0.8)
        axes[1, 0].set_title('3-gram Distribution Comparison')
        axes[1, 0].set_ylabel('Metric Value')
    
    # Run length comparison
    gt_run_0 = gt_stats['run_length_stats']['0']['mean']
    gt_run_1 = gt_stats['run_length_stats']['1']['mean']
    ext_run_0 = ext_stats['run_length_stats']['0']['mean']
    ext_run_1 = ext_stats['run_length_stats']['1']['mean']
    
    run_types = ['0-runs', '1-runs']
    gt_runs = [gt_run_0, gt_run_1]
    ext_runs = [ext_run_0, ext_run_1]
    
    x = np.arange(len(run_types))
    axes[1, 1].bar(x - width/2, gt_runs, width, label='Ground Truth', alpha=0.8)
    axes[1, 1].bar(x + width/2, ext_runs, width, label='Extracted', alpha=0.8)
    axes[1, 1].set_title('Mean Run Length Comparison')
    axes[1, 1].set_xlabel('Run Type')
    axes[1, 1].set_ylabel('Mean Length')
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels(run_types)
    axes[1, 1].legend()
    
    plt.tight_layout()
    plt.savefig(output_dir / 'functional_equivalence_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"📊 Functional equivalence visualizations saved to {output_dir}")


def main():
    # Load FSMs
    gt_path = Path("domain_machines/seven_state_human/seven_state_human/seven_state_human.machine.json")
    ext_path = Path("results/cssr_enhanced_extraction/cssr_enhanced_results.json")
    output_dir = Path("results/functional_equivalence_analysis")
    
    print("🧪 FUNCTIONAL EQUIVALENCE ANALYSIS")
    print("=" * 60)
    
    with open(gt_path, 'r') as f:
        gt_fsm = json.load(f)
    
    with open(ext_path, 'r') as f:
        ext_fsm = json.load(f)
    
    # Create simulators
    print("🔧 Setting up FSM simulators...")
    gt_simulator = FSMSimulator(gt_fsm, "ground_truth")
    ext_simulator = FSMSimulator(ext_fsm, "extracted")
    
    # Generate test sequences
    num_sequences = 1000
    sequence_length = 1000
    print(f"🎲 Generating {num_sequences} sequences of length {sequence_length} from each FSM...")
    
    # Generate from ground truth
    gt_sequences = []
    for i in range(num_sequences):
        seq = gt_simulator.generate_sequence(sequence_length, seed=i)
        gt_sequences.append(seq)
    
    # Generate from extracted FSM
    ext_sequences = []
    for i in range(num_sequences):
        seq = ext_simulator.generate_sequence(sequence_length, seed=i)
        ext_sequences.append(seq)
    
    print("✅ Sequence generation complete!")
    
    # Compute statistics
    print("📊 Computing statistical properties...")
    gt_stats = compute_sequence_statistics(gt_sequences)
    ext_stats = compute_sequence_statistics(ext_sequences)
    
    # Compare distributions
    print("🔍 Comparing statistical distributions...")
    comparisons = compare_distributions(gt_stats, ext_stats, "Ground Truth", "Extracted")
    
    # Results
    print("\n" + "🎯 FUNCTIONAL EQUIVALENCE RESULTS" + "\n" + "=" * 60)
    
    # Symbol distribution
    symbol_diff = comparisons['symbol_distribution']['mean_diff']
    print(f"📈 Symbol Distribution:")
    print(f"   Ground Truth: 0={gt_stats['symbol_distribution'].get('0', 0):.4f}, 1={gt_stats['symbol_distribution'].get('1', 0):.4f}")
    print(f"   Extracted:    0={ext_stats['symbol_distribution'].get('0', 0):.4f}, 1={ext_stats['symbol_distribution'].get('1', 0):.4f}")
    print(f"   Mean difference: {symbol_diff:.4f}")
    
    # Transition probabilities
    trans_diff = comparisons['transition_probabilities']['mean_diff']
    print(f"\n🔄 Transition Probabilities:")
    print(f"   Mean difference: {trans_diff:.4f}")
    print(f"   Max difference: {comparisons['transition_probabilities']['max_diff']:.4f}")
    
    # N-gram analysis
    print(f"\n📝 N-gram Pattern Analysis:")
    for n in range(2, 5):
        ngram_key = f'{n}gram'
        if ngram_key in comparisons['ngram_comparisons']:
            comp = comparisons['ngram_comparisons'][ngram_key]
            print(f"   {n}-grams: Mean diff={comp['mean_diff']:.4f}, KL div={comp['kl_divergence']:.4f}")
            print(f"            Unique patterns: GT={comp['unique_patterns_1']}, Ext={comp['unique_patterns_2']}")
    
    # Run length analysis  
    print(f"\n🏃 Run Length Analysis:")
    gt_0_run = gt_stats['run_length_stats']['0']['mean']
    ext_0_run = ext_stats['run_length_stats']['0']['mean']
    gt_1_run = gt_stats['run_length_stats']['1']['mean']
    ext_1_run = ext_stats['run_length_stats']['1']['mean']
    
    print(f"   0-runs: GT={gt_0_run:.2f}, Ext={ext_0_run:.2f}, Diff={abs(gt_0_run-ext_0_run):.2f}")
    print(f"   1-runs: GT={gt_1_run:.2f}, Ext={ext_1_run:.2f}, Diff={abs(gt_1_run-ext_1_run):.2f}")
    
    # Overall assessment
    print(f"\n🏆 OVERALL FUNCTIONAL EQUIVALENCE:")
    
    # Create composite score
    composite_score = (
        symbol_diff * 2 +  # Weight symbol distribution highly
        trans_diff * 2 +   # Weight transition probabilities highly  
        comparisons['ngram_comparisons'].get('3gram', {}).get('mean_diff', 0) +
        abs(gt_0_run - ext_0_run) / 10 +  # Normalize run length differences
        abs(gt_1_run - ext_1_run) / 10
    ) / 6
    
    print(f"   Composite difference score: {composite_score:.4f}")
    
    if composite_score < 0.05:
        print("   ✅ EXCELLENT: FSMs are functionally equivalent!")
    elif composite_score < 0.1:
        print("   ✅ GOOD: FSMs are very similar functionally!")
    elif composite_score < 0.2:
        print("   ⚠️  MODERATE: FSMs have similar but not identical behavior")
    else:
        print("   ❌ POOR: FSMs generate significantly different sequences")
    
    # Create visualizations
    create_comparison_visualizations(gt_stats, ext_stats, comparisons, output_dir)
    
    # Save detailed results
    results = {
        'ground_truth_stats': gt_stats,
        'extracted_stats': ext_stats,
        'comparisons': comparisons,
        'composite_score': float(composite_score),
        'assessment': {
            'symbol_distribution_diff': float(symbol_diff),
            'transition_probability_diff': float(trans_diff),
            'run_length_diffs': {
                '0_runs': float(abs(gt_0_run - ext_0_run)),
                '1_runs': float(abs(gt_1_run - ext_1_run))
            }
        }
    }
    
    with open(output_dir / 'functional_equivalence_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print("=" * 60)
    print(f"📁 Detailed results saved to {output_dir}")


if __name__ == '__main__':
    main()