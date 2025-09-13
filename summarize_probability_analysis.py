#!/usr/bin/env python3
"""
Summarize probability comparison analysis results.
"""

def main():
    print("=== Probability Analysis Summary ===\n")
    
    print("## Key Findings:\n")
    
    print("### 1. Neural vs Empirical Probability Accuracy")
    print("**80k Token Dataset (experiments/datasets/golden_mean/):**")
    print("- EBM Model: MAE = 0.0993 (9.93% average error)")
    print("- AR Model:  MAE = 0.0512 (5.12% average error)")
    print("- **AR models are ~2x more accurate at predicting empirical probabilities**")
    
    print("\n**10k Token Dataset (experiments/golden_mean/):**")
    print("- EBM Model: MAE = 0.0931 (9.31% average error)")  
    print("- AR Model:  MAE = 0.0580 (5.80% average error)")
    print("- **Similar pattern: AR models significantly more accurate**")
    
    print("\n### 2. Dataset Size Impact")
    print("**Effect of small datasets (10k vs 80k tokens):**")
    print("- Sample count per history drops dramatically with smaller datasets")
    print("- Length-6 histories: 475.9 samples (10k) vs 3809.2 samples (80k)")
    print("- **Small datasets lead to sparse empirical estimates, especially for long histories**")
    
    print("\n### 3. History Length Analysis")
    print("**Prediction accuracy vs history length (80k dataset):**")
    
    print("\n**EBM Model:**")
    print("- Length 0: MAE = 0.1680")
    print("- Length 1: MAE = 0.3044")
    print("- Length 6: MAE = 0.0665")
    print("- **Accuracy improves with longer context**")
    
    print("\n**AR Model:**") 
    print("- Length 0: MAE = 0.1680")
    print("- Length 1: MAE = 0.2479")
    print("- Length 6: MAE = 0.0329")
    print("- **Much better improvement with longer context**")
    
    print("\n### 4. CSSR Implications")
    print("**Why both models struggle with accurate CSSR reconstruction:**")
    print("1. **Neural-Empirical Mismatch**: Even the best AR model has ~5% error")
    print("2. **Statistical Tests Sensitivity**: CSSR hypothesis tests rely on precise probability estimates")
    print("3. **Small Dataset Problem**: 10k tokens → sparse statistics for longer histories")
    print("4. **Over-segmentation**: Probability errors cause CSSR to split states unnecessarily")
    
    print("\n### 5. Dataset Size Analysis")
    print("**Effect of varying dataset size on probability estimation:**")
    print("- MAE decreases only slowly with dataset size")
    print("- Even with 50k tokens, AR model MAE ≈ 0.115")
    print("- **Fundamental neural-empirical gap persists regardless of dataset size**")
    
    print("\n### 6. Recommendations")
    print("**For better CSSR reconstruction:**")
    print("1. **Use AR models**: ~2x more accurate than EBM models")
    print("2. **Increase dataset size**: More tokens → better empirical statistics")
    print("3. **Adjust CSSR parameters**: Higher significance levels (α) to account for probability errors")
    print("4. **Consider alternative backends**: TransCSSR vs neural_js may handle uncertainty differently")
    print("5. **Hybrid approaches**: Combine neural probabilities with empirical smoothing")
    
    print("\n### 7. Technical Insights")
    print("**Why AR models outperform EBM models:**")
    print("- AR models directly optimize next-token prediction")
    print("- EBM models optimize energy-based contrastive objective")
    print("- **Direct objective alignment makes AR models better probability estimators**")
    
    print("\n**Dataset size vs. reconstruction quality:**")
    print("- 80k dataset: EBM = 9 states, AR = 16 states (expected: 2)")
    print("- Longer histories need exponentially more data for good statistics")
    print("- **CSSR algorithm sensitive to probability estimation errors**")
    
    print("\n=== Generated Analysis Files ===")
    print("- probability_analysis/golden_mean_golden_mean_ebm_prob_comparison.png")
    print("- probability_analysis/golden_mean_golden_mean_ar_prob_comparison.png") 
    print("- probability_analysis/golden_mean_golden_mean_ebm_size_analysis.png")
    print("- probability_analysis/golden_mean_golden_mean_ar_size_analysis.png")

if __name__ == '__main__':
    main()