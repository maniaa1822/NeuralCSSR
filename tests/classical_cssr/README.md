# Classical CSSR Tests

This folder contains test scripts for analyzing and debugging the classical CSSR implementation.

## Test Scripts

- **`test_classical_cssr.py`** - Main test script for classical CSSR on generated dataset
- **`analyze_cssr.py`** - Detailed analysis of CSSR behavior and state formation
- **`debug_chi_square.py`** - Debug chi-square statistical test implementation
- **`test_cssr_params.py`** - Test different CSSR parameters (significance levels, min counts)
- **`test_kl_params.py`** - Test KL divergence thresholds for optimal state recovery

## Key Findings

**Best Classical CSSR Configuration:**
- **Test Type**: KL divergence 
- **Min Count**: 2
- **KL Threshold**: 0.01
- **Results**: 8 states discovered, 70.85% accuracy, 37.19% coverage

**Ground Truth Recovery:**
- **Exact 3 states**: min_count=2, KL_threshold=0.30 → 63.29% accuracy
- **Chi-square tests**: Too conservative, always merge to 1 state
- **KL divergence**: More sensitive, better state discrimination

## Usage

Run from project root:
```bash
# Main test
uv run python tests/classical_cssr/test_classical_cssr.py

# Detailed analysis  
uv run python tests/classical_cssr/analyze_cssr.py

# Parameter tuning
uv run python tests/classical_cssr/test_kl_params.py
```

These tests established the classical CSSR baseline for comparison with neural approaches.