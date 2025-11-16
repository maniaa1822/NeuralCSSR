# Discovering Causal States with Neural Networks: A Two-Stage CSSR Algorithm

## Introduction

Imagine you're observing a complex system that generates a sequence of binary symbols (0s and 1s). Your task is to discover the minimal set of "states" that capture all the predictive information about the system's future behavior. This is the problem of **epsilon machine discovery** - finding the simplest causal model that explains observed data.

The classical solution to this problem is **CSSR** (Causal State Splitting Reconstruction), an algorithm that discovers epsilon machines from data by grouping histories with similar predictive properties. But what if we could harness the power of modern neural networks - specifically, transformers like GPT - to estimate the probability distributions needed for this discovery process?

This blogpost explores our **two-stage CSSR algorithm**, which combines neural probability estimation with information-theoretic clustering to discover minimal causal states from sequences. We'll dive into the theory behind the algorithm, explain why it works, and show how it successfully recovers epsilon machines for both simple and complex processes.

## Background: Epsilon Machines and Causal States

### What is an Epsilon Machine?

An **epsilon machine** is a mathematical object that represents the minimal causal structure of a stochastic process. It consists of:

- **States** (ε-states or causal states): Groups of histories that are indistinguishable in terms of their predictive properties
- **Transitions**: How the system moves between states based on observed symbols
- **Emissions**: The probability distribution over next symbols for each state

The key insight is that two histories belong to the same causal state if and only if they make identical predictions about the future, regardless of how different their pasts might be.

### Example: The Golden Mean Process

Consider the **Golden Mean** process, which generates binary sequences with one rule: *no two consecutive 0s are allowed*. This simple constraint gives rise to two causal states:

- **State A**: Last symbol was 0 → Must emit 1 next (P(1)=1.0)
- **State B**: Last symbol was 1 → Can emit either 0 or 1 (P(0)=0.5, P(1)=0.5)

Despite infinitely many possible histories (0, 1, 10, 11, 101, 110, ...), they all collapse into just these two causal states based on their predictive properties.

## The Challenge: Neural Probability Estimation

Traditional CSSR uses empirical frequency counts to estimate probabilities like P(next symbol | history). This works well for simple processes but faces the **curse of dimensionality**: as history length grows, the number of possible histories grows exponentially, making frequency-based estimation impractical.

**Neural networks**, particularly transformer models like nanoGPT, offer a powerful alternative. By training a character-level language model on binary sequences, we obtain a **neural probability estimator** that can:

1. Generalize across similar histories
2. Handle long contexts efficiently
3. Estimate P(next k symbols | history) through autoregressive rollout

However, using neural probabilities introduces new challenges:
- **Calibration**: Neural logits may not be perfectly calibrated probabilities
- **Noise**: Model uncertainty can cause histories from the same true state to have slightly different estimated distributions
- **Computational cost**: Computing full pairwise comparisons is O(n²) in the number of histories

Our two-stage algorithm addresses all these challenges.

## The Two-Stage Algorithm

The algorithm consists of two main stages that progressively refine our understanding of causal states:

### Stage A: Emission Clustering

**Goal**: Group histories that have similar immediate next-symbol predictions.

**Method**: Agglomerative hierarchical clustering based on Jensen-Shannon (JS) divergence between emission distributions.

**Algorithm**:
1. For each sampled history h, compute the emission distribution: **P(x | h)** where x ∈ {0, 1}
2. Start with each history as its own singleton cluster
3. Iteratively merge the closest pair of clusters based on centroid JS divergence:
   - Compute JS divergence between cluster centroids (mean emission distributions)
   - Merge the pair with minimum JS divergence
   - Recompute centroid for the merged cluster
4. Stop when the minimum JS divergence exceeds threshold τ_A

**Why it works**: Histories from the same causal state must have identical emission distributions (by definition of epsilon machines). Stage A groups histories into "emission buckets" - each bucket contains histories with similar P(next | h).

**Key insight**: We use **centroid-based clustering** rather than all-pairwise comparisons. The centroid (mean emission) represents the "typical" emission for a cluster, allowing us to merge clusters efficiently while tolerating small neural estimation noise.

### Stage B: Conditional JS Refinement

**Goal**: Within each emission bucket, distinguish states that have the same immediate predictions but different long-term behavior.

**Why it's needed**: Two histories might emit the same next symbol with the same probabilities, but diverge in their predictions for future steps. For example:
- History h₁: P(0)=0.5, and if 0 is emitted, future behavior follows pattern A
- History h₂: P(0)=0.5, but if 0 is emitted, future behavior follows pattern B

Stage A would group these together (same emission), but they belong to different causal states (different successor kernels).

**Method**: Use **conditional JS divergence** on k-step rollout distributions.

**Algorithm**:
1. For each emission bucket from Stage A:
   - Select a small set of representative histories (to avoid O(n²) comparisons)
   - Compute k-step rollout distributions for each representative
   - Cluster representatives using conditional JS divergence
   - Assign all histories in the bucket to their nearest representative cluster

**The Theory of Conditional JS Divergence**:

Traditional JS divergence on k-step distributions has a problem: it mixes together two sources of variation:
1. **Marginal differences**: P(first symbol | h₁) ≠ P(first symbol | h₂)
2. **Conditional differences**: P(future | first symbol, h₁) ≠ P(future | first symbol, h₂)

Since Stage A already grouped by marginal (emission) similarity, we want Stage B to focus on conditional differences - the **successor kernel** structure.

**Conditional JS divergence** removes the marginal component by conditioning on the first symbol:

```
JS_cond(h₁, h₂; k) = w̄₀ · JS(P(next k-1 | 0, h₁), P(next k-1 | 0, h₂))
                    + w̄₁ · JS(P(next k-1 | 1, h₁), P(next k-1 | 1, h₂))
```

where:
- w̄ₐ = ½[P(a | h₁) + P(a | h₂)] is the average marginal probability for symbol a
- P(next k-1 | a, h) is the conditional distribution over the next k-1 symbols given that symbol a was emitted first

**Why this works**:
1. By conditioning on the first symbol, we remove the mixture effect from marginal differences
2. We focus solely on whether the **successor kernels** are the same after branching on each symbol
3. The weighted average respects the natural marginal distribution while comparing conditionals
4. This is exactly the information we need to distinguish causal states that have the same emission but different transition structures

### Stage C: State Remerging (Optional)

For processes with **infinite memory** (like the Even Process, where you need to track the parity of arbitrarily long runs of 1s), Stage B might still over-split states due to:
- Different history lengths representing the same functional state
- Neural estimation noise causing slight variations

**Stage C** performs a final remerging pass:
1. Compare all discovered states pairwise using both:
   - Emission JS divergence (should be nearly identical)
   - Multi-step rollout JS divergence (should be nearly identical)
2. Merge states that are functionally equivalent within tight thresholds

This is critical for recovering the correct 2-state structure of infinite-memory processes.

## Optional Enhancement: Backward Stability

**The Problem**: For processes with long memory, many different long histories might belong to the same causal state. For example, in the seven-state human machine:
- "001001ba" and "ba" might be the same state
- The prefix "001001" is irrelevant for prediction

**Backward Stability** finds the **minimal sufficient suffix** for each history:
1. Start with the full history h
2. Try progressively shorter suffixes: h[-L+1:], h[-L+2:], ..., h[-Lmin:]
3. For each suffix, compute P(next | suffix) and compare to P(next | full history) using JS divergence
4. Find the shortest suffix where JS divergence stays below tolerance τ_bs
5. This is the minimal suffix that preserves predictive information

**Why it helps**:
- Reduces redundancy by mapping multiple long histories to the same minimal representative
- Makes clustering more robust by removing irrelevant prefix information
- Helps identify the true memory structure of the process

**Applied within Stage A**: After emission clustering, we apply backward stability within each bucket to find minimal contexts, then use these for Stage B refinement.

## Implementation Details

### Sampling Strategy

To avoid analyzing all possible histories (exponentially many), we use **emission-stratified sampling**:
1. Sample a large pool of histories from the dataset
2. Compute emission P(0 | h) for each
3. Divide into strata based on P(0) quantiles
4. Sample uniformly from each stratum
5. Deduplicate to ensure diverse representatives

This ensures coverage across the emission spectrum while maintaining tractable computation.

### Caching K-Step Distributions

Computing k-step rollouts is expensive (requires 2^k neural forward passes via tree expansion). We use aggressive caching:
- Cache all k-step distributions by history tuple
- Reuse cached distributions across clustering iterations
- Typical cache hit rates: 80-95%

### Platt Calibration

Neural networks often produce poorly calibrated probabilities. We apply **Platt calibration**:
1. Fit logistic regression on validation data: `P_calibrated(1) = σ(a·logit(P_neural(1)) + b)`
2. Apply calibration transform to all emission and rollout probabilities
3. This improves stability of JS divergence computations

## Results and Examples

### Golden Mean Process

**Ground Truth**: 2 states (A: last=0, B: last=1)

**Training**: nanoGPT with context window=64, 5000 iterations
- Validation loss: 0.666 bits/symbol (matches theoretical entropy)

**Discovery** (Stage A only, no Stage B needed):
```bash
python cssr_discovery/unsupervised_fast_original.py \
  --preset golden_mean \
  --stage_a_threshold 0.001 \
  --n_samples 100 --L 5
```

**Result**: **2 states discovered**
- State 0: P(0)=0.00, P(1)=1.00 (last was 0, must emit 1)
- State 1: P(0)=0.50, P(1)=0.50 (last was 1, can emit either)
- Weighted purity: 1.00
- Exact recovery of ground truth states

### Even Process

**Ground Truth**: 2 states (E: even number of 1s in current run, O: odd number)

**Challenge**: Infinite memory (need to track parity of arbitrarily long runs)

**Training**: nanoGPT with context window=128, 10k iterations
- Validation loss: 0.671 bits/symbol (near theoretical 2/3)

**Discovery** (with Stage C remerging):
```bash
python cssr_discovery/unsupervised_fast_original.py \
  --preset even_process \
  --stage_a_threshold 0.001 \
  --stage_b_threshold 0.001 \
  --k_refine 4 --enable_remerging \
  --n_samples 100 --L 6
```

**Result**: **2 states discovered** (after remerging from initial 4-6 states)
- State E: P(0)=0.50, P(1)=0.50 (even run, can emit 0 or continue with 1)
- State O: P(0)=0.00, P(1)=1.00 (odd run, must continue with 1 to make even)
- Stage C successfully merged histories of different lengths that represent same functional state

### Seven-State Human Machine

**Ground Truth**: 7 states with complex suffix structure
- States: {bb, aaa, aaab, ba, bab, baab, baa}
- Memory length ≤ 4 symbols

**Training**: nanoGPT with context window=64, 10k iterations
- Validation loss: near theoretical entropy

**Discovery** (with backward stability):
```bash
python cssr_discovery/unsupervised_fast_original.py \
  --preset seven_state_human_char_large \
  --backward_stability --tolerance_bits 1e-3 --min_suffix_len 2 \
  --stage_a_threshold 0.001 --n_samples 100 --L 5
```

**Result**: **7 states discovered**
- Backward stability identified minimal suffixes (e.g., "ba" instead of "001001ba")
- Weighted purity: 0.95+
- Successfully distinguished all 7 causal states despite complex structure

## Key Insights and Takeaways

### Why Two Stages?

1. **Efficiency**: Stage A uses simple emission distributions (2 probabilities), avoiding expensive k-step rollouts until Stage B refinement
2. **Robustness**: Emission clustering is more stable under neural estimation noise
3. **Scalability**: Representative-based Stage B reduces O(n²) to O(k) comparisons per bucket

### Why Conditional JS?

- Standard JS on rollouts conflates marginal and conditional differences
- Since epsilon machine states are defined by **identical successor kernels**, we need to compare conditionals directly
- Conditional JS isolates the true structural differences after accounting for emission similarity

### When to Use Backward Stability?

- **Use it** for processes with long but finite memory (e.g., seven-state machine)
- **Use it** when you see many long histories that likely belong to the same state
- **Skip it** for processes with very short memory (e.g., golden mean with L=1)

### When to Use Stage C Remerging?

- **Essential** for infinite memory processes (even process)
- **Helpful** when Stage B over-splits due to neural noise
- Use tight thresholds (1e-4 for emissions, 5e-4 for rollouts) to avoid incorrect merges

### Hyperparameter Guidelines

| Parameter | Typical Range | Effect |
|-----------|---------------|--------|
| `stage_a_threshold` | 0.001-0.025 | Lower = more emission groups (finer initial split) |
| `stage_b_threshold` | 0.001-0.02 | Lower = more final states (less merging) |
| `k_refine` | 3-4 | Higher = captures longer-range dependencies (slower) |
| `tolerance_bits` | 1e-3 - 1e-2 | For backward stability; lower = stricter suffix minimality |
| `n_samples` | 50-100 | More samples = better coverage (slower) |

## Theoretical Foundations

### Information-Theoretic View

The epsilon machine captures the **minimal sufficient statistic** for prediction. Our algorithm operationalizes this through:

1. **Sufficiency**: Histories in the same cluster have the same predictive distributions
   - Verified by JS divergence (information-theoretic distance)
   - Both immediate (emissions) and long-term (k-step rollouts)

2. **Minimality**: We aggressively merge until information loss exceeds threshold
   - Hierarchical clustering finds natural information-preserving boundaries
   - Backward stability removes redundant prefix information

3. **Causality**: Conditional JS focuses on causal structure (successor kernels)
   - Removes non-causal marginal variations
   - Aligns with epsilon machine definition: same state ⟺ same future distribution

### Connection to Classical CSSR

Classical CSSR:
- Uses empirical frequencies for P(x | h)
- Tests statistical significance via χ² or likelihood ratio tests
- Grows a tree of histories, splitting when tests reject equality

Neural two-stage CSSR:
- Uses neural estimates for P(x^k | h)
- Tests information-theoretic similarity via JS divergence
- Merges clusters hierarchically until information divergence exceeds threshold

Both aim for the same goal (epsilon machine discovery), but neural approach:
- Handles longer contexts through learned representations
- Trades statistical rigor for generalization power
- Better suited for complex processes where frequency-based estimation fails

## Computational Complexity

### Classical Pairwise Approach: O(n² · 2^k)
- Compare all n histories pairwise: O(n²)
- Each comparison requires k-step rollout: O(2^k) neural forward passes
- Intractable for n > 100

### Two-Stage Approach: O(n² + m · r · 2^k)
- Stage A: O(n²) emission comparisons (cheap, just 2 probabilities each)
- Stage B: O(m · r² · 2^k) where m = # buckets, r = # representatives per bucket
  - Typically m ≈ 5-10, r ≈ 5-10
  - Reduction: ~1000× fewer expensive rollout comparisons
- Practical for n = 100-1000 histories

### Caching Benefit
- Without cache: repeat rollouts for same histories across iterations
- With cache: ~80-95% hit rate
- Effective speedup: 5-20×

## Future Directions

1. **Adaptive thresholding**: Learn optimal τ_A and τ_B from data
2. **Active learning**: Intelligently sample histories that resolve ambiguities
3. **Uncertainty quantification**: Use neural ensembles or Bayesian approaches to measure confidence in state assignments
4. **Scaling to larger alphabets**: Extend beyond binary to arbitrary symbol sets
5. **Online discovery**: Incrementally update epsilon machine as new data arrives

## Conclusion

The two-stage CSSR algorithm successfully bridges classical information theory with modern neural estimation. By carefully separating emission clustering (Stage A) from successor kernel refinement (Stage B), and using conditional JS divergence to focus on causal structure, we achieve:

- **Accurate recovery** of ground truth epsilon machines across diverse processes
- **Computational efficiency** through representative-based comparisons and caching
- **Robustness** to neural estimation noise via hierarchical clustering
- **Flexibility** to handle both finite and infinite memory processes

The key theoretical insight is that **conditional JS divergence on multi-step rollouts** captures exactly the information needed to distinguish epsilon machine states, while **emission-based initial clustering** provides an efficient coarse-graining that makes the refinement tractable.

This work demonstrates that neural networks can serve not just as black-box predictors, but as powerful tools for **scientific discovery** - extracting interpretable causal structure from complex sequential data.

---

## References and Further Reading

- **Epsilon machines**: Shalizi, C. R., & Crutchfield, J. P. (2001). "Computational mechanics: Pattern and prediction, structure and simplicity"
- **Classical CSSR**: Shalizi, C. R., & Klinkner, K. L. (2004). "Blind construction of optimal nonlinear recursive predictors for discrete sequences"
- **Jensen-Shannon divergence**: Lin, J. (1991). "Divergence measures based on the Shannon entropy"
- **Neural language models**: Radford, A. et al. (2019). "Language Models are Unsupervised Multitask Learners"
- **This implementation**: See `cssr_discovery/unsupervised_fast_original.py` in the NeuralCSSR repository

## Appendix: Complete Example

Here's a complete end-to-end example for the Even Process:

```bash
# Step 1: Generate dataset
uv run python pysm_generator.py --machine even_process --length 50000 \
  --output experiments/datasets --seed 42

# Step 2: Prepare nanoGPT data
uv run --with numpy python nanoGPT/data/even_process/prepare.py

# Step 3: Train nanoGPT model
cd nanoGPT
uv run --with torch --with numpy python train.py \
  config/train_even_process_char.py \
  --device=cuda --block_size=128 --dropout=0.0 \
  --max_iters=10000 --lr_decay_iters=10000 \
  --out_dir=out-even-process --always_save_checkpoint=True
cd ..

# Step 4: Discover epsilon machine
uv run --with torch python cssr_discovery/unsupervised_fast_original.py \
  --preset even_process \
  --stage_a_threshold 0.001 --stage_b_threshold 0.001 \
  --k_refine 4 --enable_remerging \
  --n_samples 100 --L 6 \
  --output_json results/even_process_results.json

# Result: 2 states discovered (E and O)
# Average loss: 0.671 bits/symbol
# Weighted purity: 1.00
```

The discovered epsilon machine correctly captures the even-parity constraint with just 2 states, despite the infinite memory nature of the process.
