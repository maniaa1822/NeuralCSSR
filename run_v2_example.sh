#!/bin/bash
# Example: Running unsupervised_fast_v2.py with backward stability

echo "========================================="
echo "Unsupervised Fast v2 - Example Usage"
echo "========================================="
echo ""
echo "This script demonstrates the ultra-fast unsupervised epsilon machine"
echo "discovery with BACKWARD STABILITY enabled by default."
echo ""

# Create results directory if it doesn't exist
mkdir -p results

# Example 1: Seven-state human (recommended settings)
echo "Example 1: Seven-state human with backward stability (default)"
echo "Command:"
echo "  uv run python cssr_discovery/unsupervised_fast_v2.py \\"
echo "    --preset seven_state_human_char_large \\"
echo "    --n_samples 200 --L 5 \\"
echo "    --output_json results/seven_state_v2.json"
echo ""

# Example 2: Even process with custom backward stability
echo "Example 2: Even process with custom backward stability settings"
echo "Command:"
echo "  uv run python cssr_discovery/unsupervised_fast_v2.py \\"
echo "    --preset even_process \\"
echo "    --n_samples 150 --L 6 \\"
echo "    --tolerance_bits 1e-3 --min_suffix_len 1 \\"
echo "    --output_json results/even_process_v2.json"
echo ""

# Example 3: Fast mode
echo "Example 3: Golden mean with fast mode"
echo "Command:"
echo "  uv run python cssr_discovery/unsupervised_fast_v2.py \\"
echo "    --preset golden_mean \\"
echo "    --n_samples 50 --L 3 --fast_mode \\"
echo "    --output_json results/golden_mean_v2.json"
echo ""

# Example 4: Disable backward stability (not recommended)
echo "Example 4: Disable backward stability (NOT recommended)"
echo "Command:"
echo "  uv run python cssr_discovery/unsupervised_fast_v2.py \\"
echo "    --preset seven_state_human_char_large \\"
echo "    --disable_backward_stability --n_samples 150 \\"
echo "    --output_json results/no_backward_stability.json"
echo ""

echo "========================================="
echo "Key Features:"
echo "========================================="
echo "✓ Backward stability (DEFAULT: ENABLED)"
echo "✓ Automatic threshold selection"
echo "✓ Smart diversity sampling"
echo "✓ Vectorized computations"
echo "✓ Unified caching system"
echo "✓ K-means++ representatives"
echo ""
echo "Run with --help for full options:"
echo "  uv run python cssr_discovery/unsupervised_fast_v2.py --help"
echo ""
