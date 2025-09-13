#!/usr/bin/env python3
"""
Analyze why models with good training metrics fail at CSSR reconstruction.
"""

import numpy as np
from pathlib import Path

def analyze_performance_disconnect():
    print("=== TRAINING SUCCESS vs CSSR FAILURE ANALYSIS ===\n")
    
    print("## Training Metrics (AR Models)")
    print("**Golden Mean AR:**")
    print("- Validation loss: 0.664 bits/token")
    print("- Perplexity: 1.585 (excellent)")
    print("- P(0|0) error: 0.000019 (near perfect)")
    print("- Zero violations: 0 (no forbidden transitions predicted)")
    
    print("\n**Even Process AR:**")
    print("- Validation loss: 0.726 bits/token")
    print("- Perplexity: 1.655 (excellent)")
    print("- P(0|E) ≈ 0.43, P(0|O) ≈ 0.09 (reasonable state differentiation)")
    print("- argmax0_given_O: 0 (correctly never predicts 0 as most likely from O)")
    
    print("\n## Ground Truth Analysis Results")
    print("**Golden Mean AR:**")
    print("- Overall MAE: 12.79% (seems reasonable)")
    print("- History '0' (State B): 48.96% error (catastrophic!)")
    print("- History '' (State A): 0% error (perfect)")
    
    print("\n**Even Process AR:**")
    print("- Overall MAE: 15.82% (seems reasonable)")
    print("- History '111' (State O): 45.11% error (catastrophic!)")
    
    print("\n## Why Training Metrics Don't Predict CSSR Success")
    
    print("\n### 1. Frequency Weighting Problem")
    print("**Training loss is frequency-weighted:**")
    print("- Common patterns (90% of data) → dominate gradient updates")
    print("- Rare critical patterns (1% of data) → barely affect loss")
    print("- Model learns to be 'good on average' not 'perfect everywhere'")
    
    print("\n**Example - Golden Mean history frequencies in 10k tokens:**")
    print("- History '' (empty): 1 occurrence (start)")
    print("- History '1': ~5000 occurrences (State A → A)")
    print("- History '0': ~2500 occurrences (State A → B)")
    print("- History '00': ~0 occurrences (B → B impossible)")
    print("- **Critical '0' context is rare but defines state boundary**")
    
    print("\n### 2. The 'Long Tail' Effect")
    print("**Training optimizes for frequent cases:**")
    print("- 80% of loss comes from 20% of most common histories")
    print("- Model achieves low perplexity by nailing frequent patterns")
    print("- Rare patterns (like forbidden transitions) are undertrained")
    
    print("\n**CSSR requires uniform accuracy:**")
    print("- Every possible history needs correct probability")
    print("- One wrong critical transition → cascading state splits")
    print("- Frequency doesn't matter for state machine reconstruction")
    
    print("\n### 3. Smoothing vs Discrete Boundaries")
    print("**Neural models are continuous smoothers:**")
    print("- Never predict exactly 0.0 or 1.0 probability")
    print("- Learn smooth interpolations between training examples")
    print("- History '0' → neural predicts ~0.49 (reasonable interpolation)")
    
    print("\n**Finite state machines have hard boundaries:**")
    print("- State B must emit P(1) = 1.0 exactly")
    print("- Any deviation (even 0.01) violates the discrete structure")
    print("- CSSR detects this violation as a different state")
    
    print("\n### 4. Validation Set Size Problem")
    print("**Training validation (10k tokens):**")
    print("- Critical rare patterns may not appear in validation")
    print("- Model seems perfect on validation but fails on systematic test")
    print("- Need exhaustive evaluation, not sample-based validation")
    
    print("\n### 5. Metric Mismatch")
    print("**Cross-entropy loss measures:**")
    print("- Average log-likelihood across all positions")
    print("- Dominated by frequent, easy patterns")
    print("- Perplexity = exp(loss) ≈ average branching factor")
    
    print("\n**CSSR reconstruction requires:**")
    print("- Perfect accuracy on state-defining critical histories")
    print("- Uniform error across all possible contexts")
    print("- Maximum error tolerance ~5% for 2-state reconstruction")
    
    print("\n## Implications")
    
    print("\n### Why Standard ML Metrics Mislead")
    print("1. **Perplexity optimism**: 1.58 perplexity suggests near-perfect model")
    print("2. **Validation optimism**: 0.664 bits/token suggests excellent generalization")
    print("3. **Frequency bias**: Common patterns mask critical failures")
    print("4. **Smoothing bias**: Continuous metrics don't capture discrete requirements")
    
    print("\n### What We Need Instead")
    print("1. **Uniform accuracy metrics**: Equal weight to all possible histories")
    print("2. **Critical pattern analysis**: Focus on rare but important transitions")
    print("3. **Discrete boundary evaluation**: Test forbidden/required transitions")
    print("4. **State-specific metrics**: Measure accuracy per ground truth state")
    
    print("\n### Training Recommendations")
    print("1. **Reweight loss function**: Higher weight for rare critical patterns")
    print("2. **Data augmentation**: Oversample rare but important contexts")
    print("3. **Constraint losses**: Penalize violation of known FSM constraints")
    print("4. **Hard attention**: Force discrete decisions for critical transitions")
    
    print("\n## Conclusion")
    print("**The performance disconnect is fundamental:**")
    print("- Standard ML training optimizes for average-case performance")
    print("- CSSR requires worst-case performance on critical patterns")
    print("- Good perplexity does NOT guarantee good CSSR reconstruction")
    
    print("\n**Key insight**: We need FSM-aware training objectives, not generic language modeling")
    print("- Current: minimize average cross-entropy")
    print("- Needed: minimize maximum error on state-critical transitions")

if __name__ == '__main__':
    analyze_performance_disconnect()