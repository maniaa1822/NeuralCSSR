# Autoregressive Transformers and Causal States

This note develops the theory story behind Neural CSSR and the 2‑stage oracle CSSR algorithm:

- how autoregressive (AR) transformers internalize **predictive state**,
- how this connects to **causal states** and **belief states** in computational mechanics, and
- why our **two-stage, oracle-driven CSSR** is a principled way to extract that structure from a neural model, in contrast to purely unsupervised clustering of representations.

The goal is to justify the research arc: *treat the transformer as an oracle, recover an epsilon-machine from it, and then study how the model’s internal representation aligns with those epsilon/belief states*.

---

## 1. Predictive modeling as learning causal states

### 1.1 Causal states as minimal sufficient statistics

Consider a stationary stochastic process generating a sequence of symbols
\(X_1, X_2, \dots\) over some alphabet \(A\) (binary in our experiments).

- A **history** is a semi-infinite past \(x^- = \dots, x_{t-2}, x_{t-1}\).
- A **predictive statistic** is a function \(\eta\) that maps histories to some representation \(\eta(x^-)\).

We are interested in statistics that are sufficient for prediction:

> A statistic \(\eta\) is **predictively sufficient** if the conditional distribution of the future given \(\eta(x^-)\) matches that given the full history:
> \[
>  P(X_{t+1:\infty} \mid \eta(X_{-\infty:t}) = \eta(x^-)) \;=\;
>  P(X_{t+1:\infty} \mid X_{-\infty:t} = x^-).
> \]

Among all such statistics, there exists a **minimal** one: the **causal state map**
\(\epsilon(x^-)\) that groups histories into equivalence classes:

- Two histories \(x^-\) and \(y^-\) are in the same **causal state** if they induce the same future distribution:
  \[
  P(X_{t+1:\infty} \mid X_{-\infty:t} = x^-) \;=\;
  P(X_{t+1:\infty} \mid X_{-\infty:t} = y^-).
  \]
- The image \(S_t = \epsilon(X_{-\infty:t})\) is the **causal state process**.

Key properties of the causal state process (epsilon-machine):

1. \(S_t\) is Markovian.
2. It is recursively updatable: \(S_{t+1} = T(S_t, X_{t+1})\).
3. The observed process is a function of the hidden state process.
4. \(\epsilon\) is minimal among all sufficient statistics (no other sufficient statistic is a “smaller” function of history).

An **epsilon-machine** is the tuple (state set, transition map, emission probabilities) associated with these causal states.

### 1.2 Autoregressive transformers as approximate causal-state predictors

Autoregressive language models (transformers, RNNs, etc.) are trained to approximate:
\[
P_\theta(X_t \mid X_{1:t-1} = x_{<t}),
\]
with parameters \(\theta\), by minimizing next-token prediction loss.

If the model is expressive enough and trained on enough data from a process with a finite-state epsilon-machine, then in the ideal limit:

- \(P_\theta(X_{t+1:\infty} \mid x_{<t})\) approaches the true predictive distribution,
- The model could, in principle, implement a **deterministic map** from histories to some internal state \(H_t\) (its hidden representation) that is a **sufficient statistic for prediction**.

Concretely, for a transformer:

- Let \(h_t = h_\theta(x_{<t})\) be its hidden state / representation at position \(t\).
- The prediction head maps \(h_t\) to logits over \(A\).
- If the model is optimal, the mapping \(x^- \mapsto h_t\) is (up to invertible transforms) equivalent to some predictive statistic \(\eta(x^-)\).

If we limit attention to **prediction**, we can think of an ideal AR transformer as learning **some representation of causal states**:

- There exists a (possibly non-linear, non-invertible) map \(g\) such that
  \[
  \epsilon(x^-) \approx g(h_\theta(x^-)).
  \]
- Histories that are causally equivalent (same future distribution) should map to hidden states that support the same predictive distribution.

This is the core thought:

> **If an autoregressive transformer is a good predictor for a process with a finite epsilon-machine, it must implicitly encode (a refinement of) the process’s causal states in its internal representation.**

Our goal is to **recover** those states (or a good approximation) from the model.

---

## 2. Mixtures, belief states, and what transformers really represent

### 2.1 Single process vs mixtures of processes

Above we assumed the data came from a **single** stationary process. In practice, an LM often trains on a **mixture** of processes:

- Sample a process index \(M\) from some prior.
- Generate a sequence from the process associated with \(M\).

For a **single process** with epsilon-machine states \(S_t\), the minimal sufficient statistic is the causal state \(S_t\).

For a **mixture of processes** with latent index \(M\), the minimal sufficient statistic for prediction is a **belief state**:

- A probability distribution over \((M, S_t^M)\):
  \[
  b_t(m, s) = P(M = m, S_t^m = s \mid X_{-\infty:t} = x^-).
  \]
- This belief state captures uncertainty over both which process generated the data and its internal state.

### 2.2 Transformers as belief-state encoders

If an AR transformer is trained on data from a mixture of processes, and we assume it is close to Bayes-optimal, then:

- It must map histories to hidden states \(h_t\) that are sufficient to reconstruct predictive distributions:
  \[
  P_\theta(X_{t+1} \mid h_t) \approx P(X_{t+1} \mid X_{-\infty:t} = x^-).
  \]
- The minimal such statistic (for the mixture) is a **belief state** \(b_t\).

Hence, at least conceptually, there exists a map:
\[
  h_t \mapsto b_t,
\]
or directly:
\[
  h_t \mapsto P(X_{t+1:\infty} \mid X_{-\infty:t} = x^-).
\]

Practically:

- Hidden states for **unambiguous** histories (where the process and state are nearly known) should correspond to **sharp belief states** (near corners of the simplex).
- Hidden states for **ambiguous** histories should correspond to **mixed belief states** (interior points in the simplex).
- Thus, in mixture settings, an LM does not simply learn a finite set of discrete causal states, but rather a **continuous manifold of belief states** over the underlying epsilon-machines.

This is important for interpretation:

- In the single-process case, we can talk about the model “learning the epsilon-machine.”
- In the mixture case, the model “learns the minimal belief-state machine” of the mixture.

---

## 3. Why naive representation clustering is not enough

Given the above, it is tempting to try to recover causal/belief states by clustering hidden states directly:

- Run the transformer on a large dataset.
- Collect hidden vectors \(h_t\).
- Cluster them (e.g., k-means, GMM, spectral clustering).
- Interpret clusters as “states.”

However, this is fundamentally underdetermined:

1. **No principled link to predictive equivalence**  
   Clustering uses a geometric similarity metric in representation space, which may or may not align with **predictive similarity**:
   - Two hidden states might be distant in Euclidean space but induce nearly identical predictive distributions.
   - Conversely, geometry might reflect aspects irrelevant to prediction (e.g., token identity, syntactic position).

2. **Permutation and mixing issues**  
   Hidden representations are only defined up to invertible transformations (e.g., permutations of neurons, layerwise transforms). Clusters may change under such transforms even if predictive behavior is unchanged.

3. **Belief manifolds**  
   In mixture settings, belief states form a **continuous manifold**, not a finite set. Forcing them into a small number of clusters discards meaningful degrees of freedom and may arbitrarily discretize mixtures.

4. **No convergence guarantees**  
   Without using the **output distribution**, there is no reason to expect clusters to converge to causal states, even with infinite data.

Conclusion:

> **We need a way to define equivalence classes of histories/contexts in terms of their predictive distributions, not just their hidden representations.**

This is exactly what CSSR does in the classical setting, and what our **oracle-driven 2-stage CSSR** does when we treat the transformer as a predictive oracle.

---

## 4. Oracle-driven two-stage CSSR: design and theory

### 4.1 Classical CSSR in brief

Classical CSSR:

- Works directly on data.
- Uses frequency estimates of conditional distributions:
  \[
  P(X_{t+1} \mid X_{t-L+1:t} = h)
  \]
  for many histories \(h\) of length up to \(L_{\max}\).
- Iteratively refines a partition of histories into states by:
  - testing whether extending a history with a symbol changes the conditional distribution significantly,
  - splitting/merging states accordingly,
  - ensuring transitions are deterministic.

Limitations:

- Needs many samples to estimate distributions over long histories (sample complexity \(O(|A|^L)\)).
- Pre-synchronization histories (short ambiguous contexts) can spawn many transient states.
- Implementation is complex; tuning tests and thresholds is tricky.

### 4.2 Our oracle-driven 2-stage variant (high level)

We assume access to a **predictive oracle**, e.g.:

- a trained transformer LM, or
- a known ground-truth epsilon-machine.

The oracle provides:

- **k-step future distributions** given a fixed-length past:
  \[
  p^{(k)}(x_{t+1:t+k} \mid h),
  \]
  where \(h\) is a history of length \(L_{\max}\).

Our 2-stage algorithm:

1. **Stage 1: cluster fixed-length histories using oracle predictions.**
   - Collect all length-\(L_{\max}\) histories from data.
   - For each history, query the oracle for k-step distributions at several horizons \(k \in \{1, 2, \dots\}\).
   - Use a divergence (e.g., JS) between these distributions to define an equivalence relation:
     - Two histories are equivalent if their oracle predictions are within a small tolerance at all horizons considered.
   - Start at \(k=1\), cluster histories; increase k and refine the partition until it stabilizes.

2. **Stage 2: compute minimal synchronizing suffixes.**
   - We only allow states to be formed from length-\(L_{\max}\) histories.
   - For each state (cluster), we examine all suffixes of its histories and find:
     - suffixes that occur *only* in that state, and
     - among those, the minimal ones under the suffix relation.
   - These are **minimal synchronizing suffixes**, which serve as canonical labels for the state.

This deliberately **breaks** from classical CSSR in two ways:

1. States come **only from fixed-length histories** (length \(L_{\max}\)). We never mint states from shorter pre-sync contexts.
2. We avoid the recursive phase and transient removal; the combination of oracle clustering + synchronizing suffixes already captures the predictive partition we care about on the observed histories.

### 4.3 Why fixed \(L_{\max}\) and horizon refinement?

The key design choices:

1. **Fixed-length pasts (\(L_{\max}\))**  
   - In many finite-state processes, there is a **sync length**: once you see a long enough suffix, you can infer the causal state.
   - Shorter histories (pre-sync) are ambiguous and correspond to **mixtures** over states.
   - Classical CSSR sometimes lets these short histories become states, which creates a proliferation of transient states.
   - By restricting candidate states to length-\(L_{\max}\) histories, we ensure:
     - every candidate state is defined at a depth where synchronization is possible,
     - pre-sync histories stay as ambiguous contexts and do not mint their own states.

2. **Horizon-by-horizon refinement**  
   - Causal states are defined by **entire future distributions**, not just next-step predictions.
   - We approximate this by looking at k-step futures for increasing \(k\).
   - Starting from \(k=1\), we cluster histories with similar 1-step futures; then at \(k=2\), we see if any clusters need to be split, and so on.
   - We stop when adding the next horizon does not change the partition.
   - This provides a **natural stopping rule**: the partition has stabilized with respect to the chosen horizons.

3. **Use of a predictive oracle**  
   - Instead of estimating empirical distributions via counts, we query a neural model or ground-truth generator.
   - This greatly reduces data requirements and improves robustness (the oracle generalizes across similar histories).

### 4.4 Why minimal synchronizing suffixes?

After clustering length-\(L_{\max}\) histories into states, we want a **human-readable, function-like** way to map histories to states.

- For each cluster (state), we look at all suffixes of its histories.
- A suffix \(u\) is **synchronizing** for state \(j\) if it only appears at the end of histories in state \(j\).
- A **minimal synchronizing suffix** is one where no proper suffix is also synchronizing for \(j\).

Properties:

- Minimal synchronizing suffixes provide **short symbolic labels** for states.
- Pre-sync suffixes (shorter than the true sync length) will almost always appear in multiple states, so they will not be synchronizing for any single state, and thus **cannot** be minimal.
- The set of minimal synchronizing suffixes acts like a **decision list**: given a history, check its suffixes from longest to shortest; the first matching minimal suffix tells you the state.

This reproduces the spirit of classical CSSR’s minimal predictive suffixes, but now:

- guided by oracle-based clustering, and
- with states restricted to length-\(L_{\max}\) contexts.

---

## 5. Connecting back to transformers and belief states

### 5.1 Transformers as oracles for Neural CSSR

In our setting:

- The oracle is a trained transformer LM on a binary process (e.g., even process, golden mean, seven-state human).
- We use the transformer to compute k-step future distributions for fixed-length pasts.
- Two-stage oracle CSSR builds an epsilon-machine **for the model**, which approximates:
  - the ground-truth epsilon-machine in the single-process case, or
  - the belief-state machine in the mixture case.

This has two immediate uses:

1. **Interpretability**  
   - We obtain a compact, discrete model (epsilon-machine) that summarizes the transformer’s predictive behavior.
   - The minimal synchronizing suffixes are interpretable symbolic descriptions of states.

2. **Probing hidden states**  
   - We can collect triples \((x_{<t}, h_t, s_t)\), where:
     - \(x_{<t}\) is the history,
     - \(h_t\) is the hidden representation,
     - \(s_t\) is the CSSR state assigned to \(x_{<t}\).
   - We can train a probe \(g(h_t) \to s_t\) to see whether the hidden representation effectively encodes the epsilon state.
   - High probe accuracy suggests that the transformer’s internal representation aligns closely with the causal/belief states discovered by CSSR.

### 5.2 Single-process vs mixture interpretations

- **Single process**  
  - The CSSR states of the transformer should match (up to isomorphism) the **ground-truth causal states**.
  - Hidden states cluster around these causal states (possibly with additional degrees of freedom).

- **Mixture of processes**  
  - The CSSR states of the transformer reflect **belief states** of the mixture.
  - Hidden states lie on a continuous manifold, but CSSR discretizes it into equivalence classes of similar predictive behavior.
  - Probing then reveals how the model organizes belief over underlying machines and their states.

This is the theoretical link:

> 2-stage oracle CSSR turns a black-box neural predictor into a **discrete predictive-state model**, which can then be related back to the model’s internal representations.

---

## 6. Summary of the theory connection

Putting it all together:

1. **Causal states** are the minimal sufficient statistics for prediction; they define an epsilon-machine.
2. **Autoregressive transformers**, when trained to optimality on a process, must internally represent some sufficient statistic for prediction:
   - Single-process case: approximately the process’s causal states.
   - Mixture case: belief states over process index and internal state.
3. **Naive clustering of hidden states** is not a reliable way to recover causal/belief states, because representation geometry is not constrained to align with predictive equivalence.
4. **Oracle-driven 2-stage CSSR**:
   - Uses the transformer as a predictive oracle.
   - Clusters fixed-length histories by k-step future distributions.
   - Derives minimal synchronizing suffixes as labels.
   - Produces an epsilon-machine that approximates the model’s predictive state space.
5. **Probing and amortization** (next steps):
   - Probe hidden states against CSSR states to quantify alignment between representation and causal/belief structure.
   - Learn amortized models that map sequences (or model representations) directly to epsilon-machines, using CSSR outputs as supervision.

This narrative ties autoregressive modeling, computational mechanics, and our two-stage algorithm into a coherent research program: **understand and reconstruct the predictive structures that transformers learn, in the language of causal and belief states.**

An additional practical knob here is **top-k expansion** in the oracle CSSR rollout:

- When computing k-step futures, we can expand only the top-k most probable next tokens and treat the remaining probability mass as a uniform “tail”.
- Sweeping `k` (e.g., 1, 2, 4, full) lets us study how much of the predictive behavior is captured by a small set of high-probability branches, and how many states are added when we model the low-probability tail in detail.
- If most predictive performance is already achieved with small `k` and a relatively stable set of states, this supports the view that AR models organize their behavior around a compact core of causal/belief states, with additional structure devoted to rare events.

---

## 7. Beyond “stochastic parrots”: what would count as evidence?

The “stochastic parrot vs reasoner” debate is partly about what internal structure AR models actually learn. In this framework, that question becomes:

> Do AR models learn reusable epsilon-/belief-state machines that capture underlying dynamics, or only surface-level statistics of the training stream?

Neural CSSR provides concrete tests that go beyond introspection:

- **Out-of-support interventions (counterfactuals)**  
  - Train a model on trajectories where certain actions are rare or absent in some states.
  - Use Env-view CSSR to see if the model’s counterfactual predictions under those unseen actions still produce the correct environment epsilon-machine.
  - Success suggests it has learned underlying dynamics, not only replayed observed action–observation pairs.

- **Structure-preserving transfer**  
  - Extract an environment epsilon-machine from one agent’s traces.
  - Use that machine as a world model for a different planner/agent (plan in the abstract state space, or train a new policy on it).
  - If the new agent performs well in the real environment without retraining the transformer, the model’s internal structure is a reusable world model, not just a mimic of the original behavior.

- **Recomposition / recombination tests**  
  - Train on mixtures of processes or tasks; recover their epsilon-machines.
  - Compose these machines in new ways (new mixtures, new initial conditions) and check whether planning or rollouts in the recovered machines match behavior in the recomposed environment.
  - A model that only memorizes the original distribution should fail on such recombinations; a learned predictive state machine should generalize.

- **Minimality and compression vs performance**  
  - Show that the recovered causal states are much smaller than raw observations or hidden vectors, yet sufficient to match predictive loss and control performance.
  - This demonstrates the model has internalized a minimal predictive structure in the computational-mechanics sense.

- **Invariance across tokenizations / agents**  
  - Change superficial tokenization (rename obs/acts, bundle tokens) or swap the agent policy, retrain, and re-extract epsilon-machines.
  - If the recovered environment machines align (up to relabeling), this indicates the model’s internal structure tracks underlying dynamics rather than surface form.

Together, these experiments would support a strong claim in the domains we study:

- In finite-state, synchronizable settings, well-trained AR models behave as **epsilon-machine learners**: their internal representations and predictions are organized around causal/belief-state machines that are reusable, compressive, and generalize under interventions and recomposition.
