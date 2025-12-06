




# Oracle CSSR (Two-Stage, Oracle-Driven, Sync-Length Aware)

This document describes the *theoretical foundations* of the two-stage, oracle‑driven CSSR‑style algorithm implemented in `two_stage_oracle_cssr.py`. The focus here is on the concepts from computational mechanics and predictive state representations, not on specific code structures or optimizations.

---

## 0. Conceptual setup

### Stochastic process and causal states

We assume an underlying stationary stochastic process over a finite alphabet $\mathcal{A}$, producing an infinite sequence
$$
X_{-\infty:\infty} = \dots, X_{-1}, X_0, X_1, X_2, \dots,\quad X_t \in \mathcal{A}.
$$

For any time $t$, write the semi-infinite past and future as
$$
\overleftarrow{x}_t = \dots, x_{t-2}, x_{t-1}, \quad \overrightarrow{x}_t = x_t, x_{t+1}, x_{t+2}, \dots
$$
and the conditional future distribution as
$$
\mathbb{P}(\overrightarrow{X}_t \mid \overleftarrow{X}_t = \overleftarrow{x}_t).
$$

**Causal states** (in the sense of computational mechanics) are equivalence classes of pasts that induce the same conditional distribution over futures:
$$
\overleftarrow{x}_t \sim_\epsilon \overleftarrow{x}'_t
\quad\Longleftrightarrow\quad
\mathbb{P}(\overrightarrow{X}_t \mid \overleftarrow{X}_t = \overleftarrow{x}_t)
=
\mathbb{P}(\overrightarrow{X}_t \mid \overleftarrow{X}_t = \overleftarrow{x}'_t).
$$
The equivalence classes under $\sim_\epsilon$ are the *causal states*; the resulting structure is the process’s $\epsilon$-machine.

Classical CSSR tries to approximate this partition directly from data using hypothesis tests and recursive state splitting. The present algorithm instead **uses an oracle** to approximate the future distributions and then reconstructs a causal-state-like partition from oracle‑induced similarities.

### Predictive oracle
The algorithm assumes access to an oracle—e.g., a neural language model or a known generative machine—that, given a finite history $h$, approximates distributions over length‑$k$ futures:
$$
q_k(\cdot \mid h) \approx \mathbb{P}(X_{t:t+k-1} = \cdot \mid \text{finite history } h).
$$

The oracle is used purely as a *predictive map* from pasts to future distributions; no assumptions are made about its internal structure beyond reasonable calibration.

### Finite histories and horizons

Because we cannot work with semi-infinite pasts or futures, the algorithm restricts to:

- A **maximum history length** $L_{\max}$: all candidate states are defined on length‑$L_{\max}$ pasts.
- A finite set of **prediction horizons** $k \in \mathcal{K}$ (e.g. $\mathcal{K} = \{1,2,3\}$).

For any observed position $t \ge L_{\max}$, define the length‑$L_{\max}$ history
$$
h_t = x_{t-L_{\max}+1:t} \in \mathcal{A}^{L_{\max}}.
$$
The algorithm works with the set of such histories appearing in data and the oracle’s predictions $q_k(\cdot \mid h_t)$.

### Synchronizability assumption

The theoretical picture assumes that the underlying process has a **finite-state, synchronizable \(\epsilon\)-machine**:

- Finite number of causal states.
- There exists a finite **synchronization length** such that, after observing a long enough suffix (a *synchronizing word*), the underlying causal state is known with probability 1.

This ensures that:

1. A finite history length \(L_{\max}\) is sufficient to approximate the causal state partition.
2. Each causal state can be labeled by one or more finite **synchronizing suffixes**.

---

## 1. Stage 1 — Oracle-induced partition on length-\(L_{\max}\) pasts

### Goal

Stage 1 approximates the causal state partition by clustering only length‑$L_{\max}$ histories according to the similarity of their oracle‑predicted future distributions. The key idea is:

-- Two histories are **equivalent** if their oracle predictions are close (under a divergence such as Jensen–Shannon) for *all* horizons $k \in \mathcal{K}$.

This yields a finite partition of the set of observed length‑$L_{\max}$ histories into candidate states.

### Oracle equivalence relation

Let $\mathcal{H}$ be the set of distinct length‑$L_{\max}$ histories observed. For each $h \in \mathcal{H}$ and each $k \in \mathcal{K}$, the oracle produces a distribution
$$
q_k(\cdot \mid h)
\quad\text{over } \mathcal{A}^k.
$$

Define a divergence $D(\cdot,\cdot)$ between distributions (in practice, Jensen–Shannon divergence). Fix a tolerance $\tau > 0$. For a given set of horizons $K \subseteq \mathcal{K}$, define
$$
h \approx_K h'
\quad\Longleftrightarrow\quad
D\bigl(q_k(\cdot \mid h),\, q_k(\cdot \mid h')\bigr) < \tau
\quad\forall k \in K.
$$

This relation induces a partition of $\mathcal{H}$ into equivalence classes. Intuitively, elements of the same class have indistinguishable oracle‑predicted futures up to the horizons in $K$.

### Iterative refinement over horizons

The algorithm grows the horizon set \(K\) and refines the partition:

1. **Initialize** with a short horizon (e.g. \(K = \{1\}\)). Partition \(\mathcal{H}\) into equivalence classes under \(\approx_{\{1\}}\).
2. **Add a longer horizon** \(k_{\text{new}}\) (e.g. 2, then 3, …). Set \(K' = K \cup \{k_{\text{new}}\}\) and refine the partition:
   - Within each current class, split histories according to the finer relation \(\approx_{K'}\).
   - Classes never merge, they only split.
3. **Stop** when adding a new horizon does not change the partition (no further splits) or when all desired horizons in \(\mathcal{K}\) have been used.

The final partition \(P\) is a set of blocks
\[
P = \{C_0, C_1, \dots, C_{M-1}\},\quad C_j \subseteq \mathcal{H},
\]
each block \(C_j\) interpreted as a **candidate causal state**. The theoretical justification is:

- If the oracle were exact and \(\mathcal{K}\) ranged over all \(k \ge 1\), equivalence under all horizons would coincide (on \(\mathcal{H}\)) with the true causal-state equivalence relation.
- In practice, finite \(\mathcal{K}\) and approximate oracle predictions yield a finite‑resolution approximation to this partition.

### Why only long histories become states

Classical CSSR constructs states from suffixes of *all* lengths up to a maximum, then relies on additional pruning (e.g., transient removal). Here:

- States are defined *only* as equivalence classes of length‑\(L_{\max}\) histories.
- Shorter contexts are not allowed to “mint” states in Stage 1.

This choice is theoretically motivated by synchronizability:

- Before synchronization, short histories may correspond to mixtures of causal states; they should not themselves be treated as states.
- Once \(L_{\max}\) is beyond the sync length, each true causal state corresponds to a subset of length‑\(L_{\max}\) histories with approximately identical future distributions.

Thus Stage 1 produces an approximation to the synchronized causal states directly, without ever introducing pre‑sync states.

---

## 2. Stage 2 — Minimal synchronizing suffixes

Stage 2 starts from the Stage‑1 partition of length‑$L_{\max}$ histories and asks:

> For each candidate state, which *shorter* suffixes uniquely identify that state?

The answer is a set of **minimal synchronizing suffixes** per state, which act as canonical labels analogous to CSSR’s minimal predictive suffixes.

### Synchronizing suffixes (conceptual definition)
Let $S = \{0,\dots,M-1\}$ index the candidate states, and suppose each observed length‑$L_{\max}$ history $h \in \mathcal{H}$ has been assigned a state label $s(h) \in S$ by Stage 1.
For any finite word $u \in \mathcal{A}^\ell$ with $1 \le \ell \le L_{\max}$, consider all length‑$L_{\max}$ histories in $\mathcal{H}$ that *end* with $u$:
$$
\mathcal{H}(u) = \{\, h \in \mathcal{H} : \text{suffix}_\ell(h) = u \,\}.
$$

The set of states compatible with suffix $u$ is
$$
S(u) = \{\, s(h) : h \in \mathcal{H}(u) \,\} \subseteq S.
$$

We call $u$ **synchronizing for state $j$** if
$$
S(u) = \{j\},
$$
i.e., whenever a length‑$L_{\max}$ history ends in $u$ (as observed in the data), it belongs to state $j$ and no other state.

This is a finite‑data analogue of the usual notion of a synchronizing word in a synchronizable \(\epsilon\)-machine: observing \(u\) suffices to determine the underlying state.

### Minimal synchronizing suffixes

There may be many synchronizing words for a given state. We are interested in those that are **minimal** under the suffix relation.

Formally, a word $u$ synchronizing for state $j$ is **minimal** if:

1. $S(u) = \{j\}$, and
2. No proper suffix $v$ of $u$ also satisfies $S(v) = \{j\}$.

The collection of all minimal synchronizing suffixes for state $j$,
$$
\mathcal{U}_j = \{u \in \mathcal{A}^{\le L_{\max}} : u \text{ is minimal synchronizing for } j\},
$$
provides a concise, redundancy‑free description of how the process synchronizes into state \(j\).

Each state \(j\) is thus labeled by a (typically small) set of short words that uniquely identify it when seen as suffixes of long histories.

### Why pre-sync histories are not states

The synchronizability assumption implies:

- For any word $u$ whose length is strictly less than the true synchronization length, there exist at least two causal states whose pasts share suffix $u$.
- Consequently, in the infinite‑data limit,
  $$
  S(u) \text{ will contain more than one state}
  $$
  for all such $u$, so $u$ cannot be synchronizing for any single state.

In our finite‑data, oracle‑based setting, this manifests as:

- Short words that appear in multiple Stage‑1 clusters never satisfy \(S(u) = \{j\}\) and therefore never appear in any \(\mathcal{U}_j\).
- Only words of effective length at least the sync length (or close to it) become synchronizing.

Thus the algorithm **automatically excludes pre‑sync contexts from the state set**: they remain ambiguous prefixes and are not promoted to causal states or state labels.

---

## 3. Relationship to classical CSSR

The two-stage oracle‑driven method is structurally related to classical CSSR but differs in several important theoretical respects:

- **Use of an oracle.**  
  Classical CSSR estimates future distributions purely from empirical frequencies and relies on asymptotics. Here, a predictive oracle approximates these distributions directly, so the state reconstruction is “oracle‑driven.”

- **Fixed history length for states.**  
  Classical CSSR constructs states from suffixes of varying lengths and then performs additional recursion/pruning to handle transient states. The oracle method instead fixes a history length \(L_{\max}\) and defines states only as equivalence classes of these long histories.

- **No explicit transient removal.**  
  Because states are constructed only from long histories whose oracle‑predicted futures are already near‑identical, and because pre‑sync contexts are never treated as states, there is no separate transient‑removal or recursive refinement phase.

- **Explicit synchronizing labels.**  
  Stage 2 explicitly recovers minimal synchronizing suffixes for each state, making synchronization behavior a first‑class object in the reconstruction rather than a side effect of the algorithm’s recursion.

Conceptually, the algorithm can be viewed as:

1. Using the oracle to approximate the causal-state equivalence relation among long histories.
2. Extracting the synchronizing words that realize those states as labeled, synchronizable predictive states.

---

## 4. Theoretical summary

- The underlying notion of state is the **causal state** from computational mechanics: an equivalence class of pasts with identical conditional future distributions.
- An oracle is used to approximate these conditional future distributions for finite histories and horizons.
- Stage 1 defines an **oracle-induced equivalence relation** on length‑\(L_{\max}\) histories via divergence thresholds over multiple prediction horizons, and forms candidate states as the resulting equivalence classes.
- Stage 2 identifies **minimal synchronizing suffixes** as the shortest words whose occurrence (as suffixes of length‑\(L_{\max}\) histories) uniquely determines a state.
- Synchronizability ensures that each true causal state admits at least one such suffix and that pre‑sync contexts cannot become state labels.
- The end result is an \(\epsilon\)-machine‑like representation: a finite set of predictive states with explicit synchronizing labels, reconstructed from data by leveraging oracle‑based predictions rather than direct long-run frequency estimates.
