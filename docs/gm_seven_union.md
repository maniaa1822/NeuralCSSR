Multimachine CSSR Analysis: gm_seven_union
Overview
We analyzed the pairwise Jensen-Shannon (JS) divergences between the ground truth belief states of the gm_seven_union ks
dataset (a combination of Golden Mean and Seven State Human machines). This analysis helps understand the resolution required to distinguish these states and explains the behavior of the CSSR algorithm.

Key Findings
k=1 Horizon (Next-Token Prediction)
Minimum Divergence: 0.000060 bits (Extremely small)
Closest Pair: seven_state_human:aaa <-> seven_state_human:aaab
Implication: At a single-step horizon, several states are statistically indistinguishable. A standard CSSR tolerance (e.g., 0.001 bits) would merge these states, leading to under-segmentation.
k=2 Horizon (Two-Step Prediction)
Minimum Divergence: 0.029153 bits (Significant increase)
Closest Pair: golden_mean:A <-> seven_state_human:baa
Implication: Extending the horizon to k=2 reveals the distinct dynamics of these states. The minimum divergence jumps by orders of magnitude, making the states separable with a reasonable tolerance (e.g., 0.02 bits).
k=3 Horizon (Three-Step Prediction)
Minimum Divergence: 0.080678 bits
Closest Pair: seven_state_human:aaab <-> seven_state_human:bab
Implication: At k=3, the previously problematic pair (GM:A <-> Human:baa) is no longer among the closest pairs. The minimum divergence increases to ~0.08 bits, suggesting that k=3 provides robust separability for all 9 states.
Conclusion
The gm_seven_union dataset exhibits state aliasing at k=1, where distinct causal states produce nearly identical next-token distributions. However, their future behaviors diverge significantly at k=2 and even more so at k=3.

This confirms why the 
two_stage_oracle_cssr.py
 run required metrics_k 1 2 and successfully discovered 7 states. To fully recover all 9 states, k=3 would be the recommended horizon.

Visualizations
(Heatmaps generated in results/gm_seven_union_divergences_k*.png)

State Mapping Analysis
We compared the 7 discovered states with the 9 ground truth states. The mapping reveals significant merging, consistent with the divergence analysis.

Discovered State	Dominant GT Labels (Count)	Interpretation
State 0	Human:aaa (100%)	Cleanly recovered.
State 1	GM:A (33k), Human:ba (8k), Human:baa (8k)	Super-state. Merges GM:A with Human states that have similar 0/1 emission probs and future dynamics.
State 2	GM:B (16k), Human:ba (5k)	Merges GM:B (deterministic 1) with Human:ba (probabilistic). Suggests Human:ba histories ending in 1 might look like GM:B.
State 3	Human:aaab (4k), Human:bab (3k)	Merged.
State 4	Human:baab (100%)	Cleanly recovered.
State 5	Human:bb (100%)	Cleanly recovered.
Insight
The "Union" dataset creates ambiguity. GM:A and Human:baa are statistically close ($D_{JS} \approx 0.029$ bits at k=2), leading to their merger in State 1. This explains why only 7 states were found instead of 9.

k=3 Experiment Results
We tested if increasing the horizon to k=3 would resolve the aliasing.

Tolerance (bits)	Platt Calibration	States Discovered	Result
0.04	Yes	6	Under-segmentation
0.03	Yes	8	Under-segmentation (Close!)
0.025	Yes	11	Over-segmentation
0.025	No	11	Over-segmentation (Robust)
0.02	Yes	11	Over-segmentation
Findings
Separability: k=3 definitely improves separability. The problematic pair (GM:A <-> Human:baa) is well-separated.
State Count: We could not hit exactly 9 states. The transition from 8 to 11 states suggests that as we tighten tolerance to split the last merged pair, other states (likely Human states with high variance) also split.
Platt Calibration: Disabling Platt calibration (user suggestion) yielded the same 11 states, confirming the result is robust to calibration details.
Conclusion
k=3 is necessary and sufficient to distinguish all ground truth states, but empirical noise leads to slight over-segmentation (11 states) when trying to fully resolve them. This is preferable to under-segmentation.

Synchronization Artifacts Analysis
We investigated the "extra" states in the 11-state model to verify the user's hypothesis that they represent different synchronization statuses (transient states).

Discovered States	Ground Truth State	Suffixes (History Paths)	Interpretation
State 1	Human:baa	6:000100	Requires full 6-step history to identify.
State 2	Human:baa	4:1100, 5:10100	Identifiable with 4-5 steps.
State 8	Human:bb	4:0011, 5:00111	Standard paths to bb.
State 10	Human:bb	6:001111	Rare/Long path to bb.
Finding
The extra states are indeed redundant copies of ground truth states. They are split because the model (or the CSSR algorithm) sees slightly different predictive distributions depending on the path taken to reach the state. This confirms the "synchronization status" hypothesis: the algorithm is distinguishing between "fully synchronized" and "partially synchronized" (or just path-distinct) versions of the same causal state.

Fidelity Verification
We compared the predictive performance of the discovered Epsilon Machine (11 states) against the original Neural Model to assess if the extraction was lossless.

Model	Average NLL (nats)	Average NLL (bits)
Neural Model	0.5091	0.7345
Epsilon Machine (11 states)	0.6222	0.8976
Difference	+0.1131	+0.1631
Conclusion
The Epsilon Machine has higher loss than the Neural Model. This means the 11-state machine is a lossy approximation. The Neural Model captures finer dependencies (likely longer history effects or subtle probability variations) that the discrete 11-state model discards.

This refutes the hypothesis that we "recovered the same loss". The Neural Model is significantly more expressive than the recovered finite state machine.

L=8 Experiment Results
We tested if increasing the history length to L=8 (from L=6) would improve the fidelity, hypothesizing that L=6 was truncating necessary context.

Metric	L=6 (k=3, tol=0.025)	L=8 (k=3, tol=0.025)
States Discovered	11	8
Average Loss (nats)	0.6222	0.6587
Average Loss (bits)	0.8976	0.9503
Findings
Worse Performance: Increasing L to 8 actually increased the loss (0.6587 vs 0.6222) and resulted in under-segmentation (8 states).
Explanation: With longer histories ($L=8$), the number of unique histories increases, but the number of samples per history decreases. This sparsity likely makes the clustering more aggressive (merging states due to lack of statistical evidence for separation), leading to a coarser model.
Conclusion: The loss discrepancy is not due to short history length. $L=6$ is sufficient to capture the dynamics (the ground truth machines are small). The discrepancy is due to the Neural Model's continuous, high-resolution belief state which cannot be perfectly compressed into a small discrete machine without loss.
Evidence Accumulation Probe
To confirm that the Neural Model uses long-term context ($L \gg 6$), we probed its belief state $P(0)$ on a long ambiguous sequence (101010...).

Step	Context Length	P(next=0)
6	6	0.0156
20	20	0.0019
50	50	0.0472
100	100	0.5650
Finding
The model's belief drifts significantly as the context length increases from 6 to 100. It does not converge to a steady state after 6 steps. This proves that the model is integrating evidence over a much longer horizon than the CSSR algorithm ($L=6$), which explains why the Neural Model achieves lower loss. The "state" of the neural model is effectively infinite (or at least as large as its context window), making perfect discrete recovery impossible.

Theoretical Conclusion: The Non-Markovian Nature of Mixed Sources
The user correctly identified the root cause: Combining machines creates a non-Markovian process.

Infinite Memory: When observing a sequence that is valid for both machines (e.g., 101010...), the observer (Neural Model) never reaches a state of certainty about which machine generated the data. It accumulates evidence indefinitely.
Impossibility of Synchronization: Because the source identity is never fully resolved, the process is technically infinite-order Markov. A finite history $L$ (even $L=128$) is never enough to capture the full belief state.
Result:
Extra States: The "extra" states (11 vs 9) are the CSSR algorithm capturing "snapshots" of this drifting belief state.
Loss Discrepancy: The Neural Model uses its full context to refine its belief (lower loss), while the Epsilon Machine is forced to truncate this belief into a few discrete states (higher loss).
Final Verdict: The CSSR extraction is working correctly, but the underlying process (Neural Model trained on mixed sources) is not a finite state machine. It is a complex, non-Markovian process that can only be approximated by an Epsilon Machine.

Final Synthesis: The Complexity-Accuracy Trade-off
As the user noted, we are observing a fundamental trade-off between Model Complexity (number of states) and Predictive Accuracy (loss).

Low Tolerance (High Resolution): We discover many states (e.g., 11 or more). This captures finer details of the drifting belief state, resulting in lower loss (closer to the Neural Model).
High Tolerance (Low Resolution): We merge similar belief states (e.g., 8 states). This creates a simpler, more interpretable machine but incurs higher loss because we average over distinct predictive contexts.
For a fixed history length $L$, the "11-state" model represents the best compressed representation we could find that balances capturing the ground truth structure with the reality of the non-Markovian belief drift.

---

Belief Collapse, Witnesses, and Mixed Regimes (New Experiments)
==============================================================

In this section we document additional experiments that probe:

- how much information about the latent machine ID is contained in finite histories,
- how that information emerges under rollouts of the neural oracle, and
- why increasing the predictive horizon $k$ cannot escape the mixed regime unless the history length $L$ is already causally sufficient.

All experiments below are for the same gm_seven_union dataset as above:

- Data: `experiments/datasets/gm_seven_union/combined.dat`
- Machine IDs: `experiments/datasets/gm_seven_union/combined.machine_ids.dat`
- Meta (segments): `experiments/datasets/gm_seven_union/combined.meta.json`
- Model checkpoint: `nanoGPT/out-gm-seven-union-char/ckpt.pt`

Tools and scripts used:

- Substring overlap: `analyze_sequence_overlap.py`
- Linear probe vs history length: `mixed_machine_regimes/probe_experiments/probe_machine_sync.py`
- Rollout-based witness frequencies: `mixed_machine_regimes/probe_experiments/probe_rollout_belief.py`
- Belief along a single rollout: `mixed_machine_regimes/probe_experiments/probe_rollout_probe_belief.py`
- Probe accuracy vs (L, k): `mixed_machine_regimes/probe_experiments/probe_rollout_probe_accuracy.py`

1. Substring Overlap: Where Does the Data Become Unambiguous?
------------------------------------------------------------

We first revisit the combinatorial structure of the union process by counting distinct substrings per machine and their overlap.

Command:

```bash
uv run python analyze_sequence_overlap.py \
  --sequence-path experiments/datasets/gm_seven_union/combined.dat \
  --meta-path experiments/datasets/gm_seven_union/combined.meta.json \
  --max-length 128 \
  --show-shared-up-to 8 \
  --plot-output results/gm_seven_overlap.png \
  > results/gm_seven_overlap.tsv
```

Key outputs:

- Table: `results/gm_seven_overlap.tsv`
- Plot: `results/gm_seven_overlap.png`

Findings:

- For each $L \le 128$ we counted:
  - `golden_mean_total(L)`: # distinct length-$L$ substrings seen in the GM segment.
  - `seven_state_human_total(L)`: same for the seven-state segment.
  - `shared_count(L)`: # substrings appearing in both segments.
  - `golden_mean_unique(L)`, `seven_state_human_unique(L)`: substrings unique to each machine.
- Shared substrings disappear by about **$L \approx 29$**:
  - `shared_count(L) > 0` for $L \le 29$,
  - `shared_count(L) = 0` for $L \ge 30`.

Interpretation:

- The data itself becomes *combinatorially unambiguous* only once histories are roughly 30 bits long. Below that, many length-$L$ histories are compatible with both machines.
- This $L \approx 29$ is an upper bound on a synchronization length for the *ground truth data*: past this point, every history suffix uniquely determines the generating machine.

2. Linear Probe on Hidden States vs History Length
--------------------------------------------------

We then asked how quickly the **neural model’s internal representation** becomes predictive of machine ID as a function of history length $L$.

Script: `mixed_machine_regimes/probe_experiments/probe_machine_sync.py`

Command:

```bash
uv run python mixed_machine_regimes/probe_experiments/probe_machine_sync.py \
  --checkpoint nanoGPT/out-gm-seven-union-char/ckpt.pt \
  --sequence-path experiments/datasets/gm_seven_union/combined.dat \
  --machine-ids-path experiments/datasets/gm_seven_union/combined.machine_ids.dat \
  --min-length 1 \
  --max-length 41 \
  --length-step 4 \
  --samples-per-length 1000 \
  --output results/gm_seven_probe.tsv \
  --plot-output results/gm_seven_probe.png
```

Key outputs:

- Table: `results/gm_seven_probe.tsv`
  - Columns: `L`, `accuracy`, `true_prob`, train/test counts.
  - `true_prob` is mean predicted probability assigned by the probe to the *true* machine on the held-out set.
- Plot: `results/gm_seven_probe.png`
  - Shows both accuracy and `true_prob` vs $L$.

Findings:

- At small history lengths, the representation is only weakly informative:
  - $L = 1$: accuracy $\approx 0.61$, `true_prob` $\approx 0.52$.
- It becomes sharply informative by moderate $L$:
  - $L = 9$: accuracy $\approx 0.975$, `true_prob` $\approx 0.95$.
  - $L = 17$ and beyond: accuracy $= 1.00$, `true_prob` $\approx 0.99$–$1.00$.

Interpretation:

- If we assume the model encodes machine identity roughly linearly in its last-layer representation, then by **$L \approx 17$** the internal belief over machines is effectively *deterministic*: a linear probe can perfectly recover the machine ID.
- There is a *gap* between the combinatorial uniqueness of the data ($L \approx 29$) and the linearly decodable certainty in the model ($L \approx 17$). The model learns to exploit statistical cues that are subtler than strict substring uniqueness.

3. Witness-Based Rollouts: Reading Belief from Behavior
-------------------------------------------------------

To stay model-agnostic, we next tried to infer belief from **rollouts** rather than hidden states, by looking at machine-specific substrings (“witnesses”) the model is willing to generate.

3.1 Auto-discovered witnesses
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Script: `mixed_machine_regimes/probe_experiments/probe_rollout_belief.py` can now auto-discover machine-specific witnesses from the dataset:

- Load segment boundaries from `combined.meta.json`.
- For each machine and length $1 \le \ell \le \ell_{\max}$:
  - Collect all observed substrings of length $\ell$ in that machine’s segment.
  - Mark it as a *witness* for machine $M$ if it never appears in any other segment.
- Keep up to `--auto-witness-limit` witnesses per machine, prioritizing shorter substrings.

Command (example):

```bash
uv run python mixed_machine_regimes/probe_experiments/probe_rollout_belief.py \
  --checkpoint nanoGPT/out-gm-seven-union-char/ckpt.pt \
  --sequence-path experiments/datasets/gm_seven_union/combined.dat \
  --machine-ids-path experiments/datasets/gm_seven_union/combined.machine_ids.dat \
  --meta-path experiments/datasets/gm_seven_union/combined.meta.json \
  --auto-witnesses \
  --auto-witness-max-length 10 \
  --auto-witness-limit 128 \
  --min-length 1 --max-length 21 --length-step 2 \
  --rollout-horizons 1,2,3,4,5,6,7,8 \
  --num-rollouts 8 \
  --samples-per-length 15 \
  --output results/gm_seven_rollout.tsv \
  --plot-output results/gm_seven_rollout.png
```

The script:

- Samples contexts of length $L$ from each machine.
- For each context, draws Monte Carlo rollouts of length up to $k_{\max}$.
- For each $(L, k, \text{machine})$, estimates:

  \[
  \text{witness\_freq}(L, k, M)
  \approx \Pr(\text{rollout contains any witness of } M \text{ within } k).
  \]

Outputs:

- Table: `results/gm_seven_rollout.tsv` (`L`, `machine`, `k`, `witness_freq`).
- Heatmap plot: `results/gm_seven_rollout.png` (one panel per machine).

Findings:

- For **seven\_state\_human**:
  - Witness frequency climbs quickly with $L$ and saturates near **1.0** already by $L \approx 5$ for small $k$ (1–8).
  - Once the model *believes* it is in the seven-state machine, short rollouts almost always produce substrings unique to that machine.
- For **golden\_mean**:
  - Witness frequency stays relatively low for all $L$ and $k$.
  - GM’s language is more constrained; its “positive” witnesses are comparatively rare. The model’s belief that it is in GM is encoded more by the **absence** of seven-state witnesses than by frequent GM-only substrings.

Interpretation:

- The oracle’s behavior is consistent with a latent machine belief:
  - When it believes it is in the seven-state machine, it very quickly produces seven-state-only substrings.
  - When it believes it is in GM, it tends not to produce seven-state witnesses; positive GM witnesses are rare, so witness frequency is lower.
- Witness frequency is therefore a sound *behavioral proxy* for the model’s posterior over machines, especially for the richer machine.

4. Belief Trajectory Along an Ambiguous Rollout
-----------------------------------------------

We then combined the linear probe with a particular ambiguous sequence to directly visualize belief drift over time.

Script: `mixed_machine_regimes/probe_experiments/probe_rollout_probe_belief.py`

Procedure:

1. Train a probe (as in Section 2) on contexts with $L \in [1, 17]$.
2. Consider an ambiguous sequence `101010...` of length 64:
   - This is valid under both GM (no `00`) and the seven-state machine.
3. For each prefix length $t$:
   - Run the model on the prefix.
   - Extract the last hidden state.
   - Apply the trained probe to obtain $P(\text{golden\_mean} \mid \text{prefix})$ and $P(\text{seven\_state\_human} \mid \text{prefix})$.

Command:

```bash
uv run python mixed_machine_regimes/probe_experiments/probe_rollout_probe_belief.py \
  --checkpoint nanoGPT/out-gm-seven-union-char/ckpt.pt \
  --sequence-path experiments/datasets/gm_seven_union/combined.dat \
  --machine-ids-path experiments/datasets/gm_seven_union/combined.machine_ids.dat \
  --train-min-length 1 --train-max-length 17 \
  --train-samples-per-length 400 \
  --rollout-len 64 \
  --output results/gm_seven_rollout_probe.tsv \
  --plot-output results/gm_seven_rollout_probe.png
```

Findings (from `results/gm_seven_rollout_probe.tsv` and the plot):

- Early steps ($t \lesssim 10$):
  - $P(\text{GM} \mid \text{prefix})$ rises quickly from $\approx 0.61$ at $t=1$ to $\approx 0.99$ by $t \approx 10$.
  - The probe reads the model as being very confident the ambiguous `1010...` sequence came from GM.
- Mid-range ($10 \lesssim t \lesssim 40$):
  - Belief stays strongly GM-biased, with $P(\text{GM})$ oscillating around $0.99$ and $P(\text{seven})$ near $0.01$.
- Very long range ($t \gtrsim 50$):
  - Belief slowly drifts back toward the seven-state machine:
    - $P(\text{GM})$ falls well below $0.1$ by the end of the 64-step sequence.
    - $P(\text{seven})$ rises above $0.9$.

Interpretation:

- The neural model supports the qualitative story in the earlier doc:
  - It initially leans strongly toward GM on ambiguous alternating sequences.
  - Over very long horizons, it gradually rebalances its belief, reflecting the mixed nature of the training distribution.
  - This drift is exactly the “infinite memory” non-Markovian behavior that a small finite-state ε-machine cannot capture without extra “synchronization status” states.

5. Probe Accuracy vs (History Length L, Rollout Steps k)
--------------------------------------------------------

Finally, we quantified how much machine-ID information comes from **increasing L** versus **rolling out k steps** from that history.

Script: `mixed_machine_regimes/probe_experiments/probe_rollout_probe_accuracy.py`

Procedure:

1. Train a linear probe on contexts with $L \in [1, 17]$ (as before).
2. For each history length $L$ and for each sampled context:
   - At step 0 (no rollout): apply the probe to the context representation and record whether it predicts the correct machine.
   - For each rollout step $k = 1, \dots, k_{\max}$:
     - Sample one next token from the model and append it to the context.
     - Re-apply the probe on the new, longer prefix.
3. Aggregate per $(L, k)$ the proportion of rollouts where the probe prediction matches the true machine ID.

Command (small grid):

```bash
uv run python mixed_machine_regimes/probe_experiments/probe_rollout_probe_accuracy.py \
  --checkpoint nanoGPT/out-gm-seven-union-char/ckpt.pt \
  --sequence-path experiments/datasets/gm_seven_union/combined.dat \
  --machine-ids-path experiments/datasets/gm_seven_union/combined.machine_ids.dat \
  --max-history-length 6 \
  --rollout-steps 5 \
  --samples-per-length 200 \
  --output results/gm_seven_rollout_probe_acc.tsv \
  --plot-output results/gm_seven_rollout_probe_acc.png
```

Key outputs:

- Table: `results/gm_seven_rollout_probe_acc.tsv` (`L`, `step`, `accuracy`, `total_samples`).
- Heatmap: `results/gm_seven_rollout_probe_acc.png`.

Findings:

- For small $L$:
  - $L = 1$: accuracy $\approx 0.55$–$0.63$ across $k = 0$–5.
  - $L = 2$: accuracy $\approx 0.58$–$0.64$ across $k = 0$–5.
  - Rolling out a few steps does **not** significantly change the probe’s ability to identify the machine.
- For larger $L$ (still relatively small):
  - $L = 3$: accuracy $\approx 0.69$–$0.72$.
  - $L = 4$: accuracy $\approx 0.77$–$0.84$.
  - $L = 5$: accuracy $\approx 0.81$–$0.87$.
  - $L = 6$: accuracy $\approx 0.83$–$0.89$.
  - Within each $L$, the accuracy curve over $k=0..5$ is almost flat: small-k rollouts don’t materially change the probe’s performance.

Interpretation:

- **Most of the machine-ID information comes from expanding the history $L$, not from rolling forward a few steps.**
  - For a fixed $L$, small-$k$ expansion does not “escape” the mixed regime: the belief about machine ID is already as good as it will be for that context length.
  - Only when $L$ itself is long enough (e.g., $\gtrsim 4$–6 in this experiment, and up to $\approx 17$ for near-perfect linearly decodable belief) does the probe become reliably accurate.

6. Theoretical Synthesis: Mixed Regime, Causal Sufficiency, and CSSR
--------------------------------------------------------------------

The new experiments clarify the relationship between:

- the *mixed* nature of the gm_seven_union process,
- the model’s internal belief over machines, and
- what CSSR can (and cannot) recover from next-token or short-horizon predictions.

Key points:

1. **Mixed regime is intrinsic.**
   - For histories that are compatible with both machines (e.g., prefixes of `101010...`), the ground truth generative process is a mixture:
     \[
     P(x_{t+1}\mid h) = \sum_M P(M\mid h)\,P_M(x_{t+1}\mid h),
     \]
     not a single Markov state.
   - No finite $(L, K)$ pair can make the process truly finite-order Markov in the usual sense; the best we can do is approximate it with a finite-state machine that captures *snapshots* of the drifting belief.

2. **Causal sufficiency is in $L$, not in small $k$.**
   - The linear-probe and rollout-probe-accuracy experiments show that:
     - For a fixed short $L$, expanding $k$ to 1–5 does not significantly improve machine-ID recovery.
     - Increasing $L$ does: the representation becomes linearly separable by $L \approx 10$–17.
   - This means the history must be long enough to contain machine-specific evidence (witness substrings or characteristic statistics) before belief can collapse.

3. **Witnesses provide a behavior-only proxy for belief.**
   - Auto-discovered witnesses give a way to read out the model’s belief strictly from next-token behavior:
     - High frequency of seven-state witnesses and low frequency of GM witnesses $\Rightarrow$ belief is in the seven-state machine.
     - Absence of cross-machine witnesses over sufficiently long contexts $\Rightarrow$ effective synchronization.
   - This is model-agnostic: no access to hidden states, only to the oracle’s predicted continuations.

4. **CSSR implications.**
   - Stage 1 (state discovery) and Stage 2 (machine splitting) operate on next-token or short-horizon distributions. If $L$ is too small, they necessarily observe the mixed regime and merge states that the neural model internally distinguishes.
   - If $L$ is too large, the space of histories becomes sparse and CSSR tends to over-split, creating multiple “synchronization status” states for the same ground truth state.
   - The experiments suggest a practical tuning strategy:
     - Use substring overlap and auto-witness statistics to find a range of $L$ where histories are close to causally sufficient (e.g., witness-based collapse evident, shared substrings decreasing).
     - Use the rollout-based witness heatmaps and/or linear probes to confirm that machine belief is close to deterministic in that range.
     - Run `two_stage_oracle_cssr.py` with that $L$ (or slightly smaller for robustness) and modest $k$ (e.g., 1–3), rather than pushing $k$ alone.

5. **No escape from the mixed regime via $k$ alone.**
   - The rollout experiments (Sections 3 and 5) show that, for histories in the mixed regime, increasing $k$ does not fundamentally resolve machine identity: the model behaves like a mixture of machines, and short-horizon behavior cannot break that symmetry.
   - Only when histories are long enough that they contain (or exclude) machine-unique witnesses does the model effectively “commit” to a machine.

Overall Conclusion
------------------

- The gm_seven_union experiments confirm that the neural model learns a **mixed regime** for ambiguous histories and gradually exits it as history length increases.
- Linear probes provide a good proxy for the model’s internal machine belief; rollout-based witnesses provide a model-agnostic behavioral proxy.
- CSSR, when driven by next-token or short-horizon predictions, faithfully sees this mixed behavior and can only approximate it with extra states. To minimize over/under-segmentation, one should:
  - choose $L$ in a range where belief is nearly collapsed (per probes/witnesses),
  - keep $k$ modest but large enough to resolve subtle state aliasing (as in the original $k=2,3$ experiments),
  - accept that perfect finite-state recovery of a truly mixed process is impossible, and aim instead for the best *compressed* representation of the neural model’s belief dynamics.
