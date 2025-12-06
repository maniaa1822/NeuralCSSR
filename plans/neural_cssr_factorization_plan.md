# Neural CSSR Factorization: Research Plan

## Executive Summary

This plan outlines the development of a method to **factorize joint world models** learned by neural networks into interpretable, modular components: an **Environment Machine** (physics/dynamics) and an **Agent Machine** (policy/intent). The approach leverages Neural CSSR (Causal State Splitting Reconstruction) with oracle-driven clustering to extract discrete ε-machines from trained Transformers.

---

## 1. Motivation and Core Hypothesis

### 1.1 The Problem

Standard sequence models (GPT-style Transformers) trained on environment-agent interaction data learn a **Joint Epsilon Machine** that conflates:
- **Environment dynamics**: $P(O_{next} \mid S_{env}, A)$
- **Agent policy**: $P(A_{next} \mid S_{env}, S_{agent})$

This entanglement makes interpretation, transfer, and safety analysis difficult.

### 1.2 The Hypothesis

By applying different **equivalence relations** during the clustering stage of Neural CSSR, we can factorize the joint machine into independent components:

| View | Equivalence Relation | Recovers |
|------|---------------------|----------|
| **Environment View** | $h_1 \sim h_2$ iff $\forall a: P(O \mid h_1, a) \approx P(O \mid h_2, a)$ | Physics/Map |
| **Agent View** | ⚠️ See Section 1.3 | Policy/Intent |

### 1.3 The Agent Equivalence Problem

**Issue**: The naive Agent View equivalence relation is:
```
h1 ~_intent h2  IFF  P(A | h1) ≈ P(A | h2)
```

This is **wrong** for goal-directed agents. It clusters by *what action is taken*, not *what the agent wants*.

**Counterexample (Grid World)**:
- History $h_1$: Agent at position 5, seeking goal 0 → $P(A=\text{left}) = 0.8$
- History $h_2$: Agent at position 3, seeking goal 8 → $P(A=\text{left}) = 0.8$
- Naive relation: $h_1 \sim h_2$ (same action distribution)
- Reality: Completely different goals!

**Why it fails**: The action distribution $P(A \mid h)$ conflates:
- The agent's **goal/intent** (latent)
- The agent's **physical state** (which determines how intent maps to action)

---

### 1.4 The Principled Solution: Residual Policy

The correct formulation is that **intent equivalence is a refinement of environment equivalence**:

```
h1 ~_intent h2  IFF  h1 ~_env h2  AND  P(A | h1) ≈ P(A | h2)
```

Or equivalently:
```
h1 ~_intent h2  IFF  s_env(h1) = s_env(h2)  AND  P(A | h1) ≈ P(A | h2)
```

**Interpretation**: Two histories represent the same intent if:
1. They are in the same physical location (environment-equivalent), AND
2. Given that same location, the agent takes the same action distribution

This is the **residual policy**: the part of the action distribution that remains after factoring out the physical context.

**Why this works**:
- At position 5: "Seek_0" → left, "Seek_8" → right. Different actions → different intent clusters.
- At position 1: "Seek_0" → stay, "Seek_8" → down. Different actions → different intent clusters.
- The intent clusters are consistent: all "Seek_0" histories cluster together *within each physical state*.

**Algorithmic consequence**: Intent discovery requires a **two-stage hierarchy**:

```
Stage 1: Discover S_env via Env-View clustering
         h1 ~_env h2 iff ∀a: P(O | h1, a) ≈ P(O | h2, a)

Stage 2: For each s ∈ S_env, discover intent modes
         Within histories mapping to s:
         h1 ~_intent h2 iff P(A | h1) ≈ P(A | h2)

Stage 3: Align intent modes across physical states
         (Optional: verify that "Seek_0 at pos 1" transitions to "Seek_0 at pos 4")
```

**Key insight**: The Agent Machine is not independent of the Environment Machine—it's a **product structure**:

$$S_{joint} = S_{env} \times S_{intent}$$

where the action distribution factorizes as:

$$P(A \mid h) = P(A \mid s_{env}(h), s_{intent}(h))$$

This product structure is what we're trying to recover.

---

### 1.5 Updated Equivalence Relations Summary

| View | Equivalence Relation | Structure | Recovers |
|------|---------------------|-----------|----------|
| **Environment** | $h_1 \sim h_2$ iff $\forall a: P(O \mid h_1, a) \approx P(O \mid h_2, a)$ | Independent | Physics/Map |
| **Intent** | $h_1 \sim h_2$ iff $s_{env}(h_1) = s_{env}(h_2)$ AND $P(A \mid h_1) \approx P(A \mid h_2)$ | Refinement of Env | Goal/Mode |
| **Joint** | $h_1 \sim h_2$ iff $P(X_{next} \mid h_1) \approx P(X_{next} \mid h_2)$ | Finest | Full state |

The Joint partition is the meet (common refinement) of Env and Intent. But because Intent is already defined as a refinement of Env, we have:

$$S_{joint} = S_{intent}$$

The factorization into $S_{env} \times S_{intent/env}$ (where $S_{intent/env}$ is the intent *given* environment) gives us the modular decomposition we want.

---

## 2. Algorithm: Two-Stage Oracle CSSR

### 2.1 Stage 1: Oracle Clustering on $L_{max}$ Histories

**Goal**: Partition length-$L_{max}$ histories into clusters approximating causal states.

**Procedure**:
1. Extract all unique length-$L_{max}$ histories from data
2. For each history $h$ and horizon $k \in \{1, 2, \ldots, K_{max}\}$:
   - Compute $P^{(k)}_h = \text{Oracle}(h, k)$ (k-step future distribution)
3. Initialize partition with $k=1$ using JS divergence threshold $\epsilon$
4. Refine partition for $k=2, 3, \ldots$ until stable
5. Output: Clusters (states) + representative histories

**Key Insight**: States are defined **only** from long histories. Pre-synchronization histories never mint states.

### 2.2 Stage 2: Synchronizing Suffix Discovery

**Goal**: Find minimal suffixes that uniquely identify each state.

**Procedure**:
1. For all $h \in H_{unique}$, for suffix lengths $1 \ldots L_{max}$:
   - Record which states each suffix appears in
2. For each state $j$:
   - Collect suffixes appearing **only** in state $j$
   - Keep minimal suffixes (no shorter suffix has same property)
3. Output: Minimal synchronizing suffixes per state

**Key Insight**: Pre-sync histories are automatically excluded—they appear in multiple states and fail the uniqueness criterion.

---

## 3. Experimental Validation Status

### 3.1 Completed Experiments

| Experiment | Description | Status | Key Finding |
|------------|-------------|--------|-------------|
| **Q2: Agency Detection** | Controlled vs Random regions | ✅ Complete | JS divergence of counterfactual actions maps agency (0.60 vs 0.0001 bits) |
| **Q3: Simple Factorization** | 2-state swap machine + 2-mode agent | ✅ Complete | Env View: ARI=1.00, Agent View: ARI=0.77 |
| **Q4: Grid World** | 9-state grid + goal-directed agent | ⚠️ Partial | Env View: ARI=1.00 (9 states), Agent View: Failed (policy is state-dependent) |
| **Mixed Machines** | gm_seven_union dataset | ✅ Complete | Discovered mixed regime, need $L \approx 17$ for belief collapse |

### 3.2 Key Findings from Mixed Machine Analysis

1. **Substring Overlap**: Histories become unambiguous at $L \approx 29$ (combinatorial)
2. **Linear Probe**: Model belief deterministic by $L \approx 17$ (statistical)
3. **Rollout Witnesses**: Behavior-only proxy for machine belief works
4. **$(L, k)$ Trade-off**: Information is in $L$, not small $k$; rollouts don't escape mixed regime

### 3.3 Identified Limitations

| Issue | Cause | Impact |
|-------|-------|--------|
| Agent View fails on Grid World | Policy is state-dependent: "Go Left" means different actions at different locations | Cannot cluster intent globally |
| Over-segmentation at tight $\epsilon$ | Neural model has continuous belief; discrete states are approximations | Extra "synchronization status" states |
| Under-segmentation at loose $\epsilon$ | Merges states with similar short-horizon behavior | Loses ground truth structure |
| Mixed regime | Combining machines creates non-Markovian process | Infinite memory, cannot perfectly recover |

---

## 4. Planned Work

### 4.1 Phase 1: Hierarchical Agent Factorization (Priority: High)

**Problem**: Agent View fails when policy is state-dependent.

**Proposed Solution**: Hierarchical Oracle CSSR

```
Algorithm: Hierarchical Agent Factorization
───────────────────────────────────────────
1. MAP DISCOVERY
   - Run Env-View Clustering → discover physical states S = {s₁, ..., sₖ}
   
2. STATE LOCALIZATION  
   - Train classifier: h → s (or use cluster centroids)
   
3. CONTEXTUALIZED POLICY ANALYSIS
   - For each state s:
     - Collect all histories Hₛ that map to s
     
4. INTENT DISCOVERY (per state)
   - Cluster Hₛ using Agent View → find local modes
   
5. MODE ALIGNMENT (across states)
   - Align modes: "Mode A in State 1" ≡ "Mode A in State 2"
     if transition probabilities are high
```

**Deliverables**:
- [ ] Implement `hierarchical_agent_cssr.py`
- [ ] Test on Q4 Grid World dataset
- [ ] Evaluate with goal-recovery ARI

### 4.2 Phase 2: Automatic Parameter Selection (Priority: High)

**Problem**: Manual tuning of $L_{max}$, $k$, and $\epsilon$ is tedious and dataset-dependent.

**Proposed Solutions**:

| Parameter | Current | Proposed Auto-Selection |
|-----------|---------|------------------------|
| $L_{max}$ | Manual | Witness-based: find $L$ where cross-machine witnesses vanish |
| $k$ | Manual | Elbow method: increase until partition stabilizes |
| $\epsilon$ | Manual | Silhouette score or gap statistic on JS distance matrix |

**Deliverables**:
- [ ] Implement `auto_select_L.py` using witness analysis
- [ ] Implement `auto_select_epsilon.py` using clustering metrics
- [ ] Integrate into `two_stage_oracle_cssr.py` as `--auto` flag

### 4.3 Phase 3: Transition Matrix and ε-Machine Export (Priority: Medium)

**Problem**: Current output is clusters + suffixes, not a runnable ε-machine.

**Proposed Implementation**:

```python
def build_epsilon_machine(partition, state_of_Lmax, alphabet):
    """
    Build explicit transitions:
      For each state j and symbol a:
        - Pick any h ∈ partition[j]
        - Form h_ext = (h + a)[-L_max:]
        - If h_ext in state_of_Lmax:
            transitions[j][a] = state_of_Lmax[h_ext]
    
    Estimate emissions:
      For each state j:
        - Pool all histories in partition[j]
        - Use oracle to estimate P(next_symbol | state=j)
    """
    ...
```

**Deliverables**:
- [ ] Implement `epsilon_machine.py` with `EpsilonMachine` class
- [ ] Add export to `.dot` (Graphviz), `.json`, and CSSR-compatible formats
- [ ] Add simulation capability: sample from the machine
- [ ] Add likelihood computation: $P(sequence \mid machine)$

### 4.4 Phase 4: Neural Map Projection (Priority: Medium)

**Goal**: Once discrete states are discovered, train a linear map from Transformer hidden states to a clean "Physics Embedding."

**Implementation**:
```python
# Train linear projection
W_env = train_linear_probe(
    X=transformer_hidden_states,  # shape (N, d_model)
    y=cluster_labels              # shape (N,)
)

# Project to physics embedding
z_env = hidden_state @ W_env.T  # shape (d_env,)
```

**Use Cases**:
- **Transfer**: Share physics knowledge across agents
- **Steering**: Modify beliefs for counterfactual testing
- **Visualization**: Plot agent's internal map

**Deliverables**:
- [ ] Implement `neural_map_projection.py`
- [ ] Visualize embeddings with t-SNE/UMAP
- [ ] Test transfer: train agent A's map, use for agent B

### 4.5 Phase 5: Scaling and Robustness (Priority: Low)

**Goal**: Test on larger, more complex environments.

**Planned Experiments**:
| Environment | States | Actions | Challenge |
|-------------|--------|---------|-----------|
| MiniGrid (6x6) | 36 | 4 | Partial observability |
| Atari (Pong) | Continuous | 3 | High-dimensional observations |
| TextWorld | Variable | Text | Symbolic actions |

**Deliverables**:
- [ ] Adapter for MiniGrid datasets
- [ ] Handling of continuous/high-dim observations (embedding first)
- [ ] Benchmark: discovered states vs ground truth

---

## 5. Technical Infrastructure

### 5.1 Current Codebase Structure

```
NeuralCSSR/
├── cssr_discovery/
│   ├── two_stage_oracle_cssr.py    # Main algorithm
│   ├── oracle_interface.py          # Model wrappers
│   └── utils.py                      # JS divergence, etc.
├── experiments/
│   ├── datasets/                     # Generated data
│   └── q2_agency.py, q3_simple.py, q4_grid.py
├── nanoGPT/                          # Training infrastructure
├── docs/                             # Documentation
│   ├── experiments_summary.md
│   └── gm_seven_union.md
└── plans/                            # This document
```

### 5.2 Needed Additions

| Component | File | Status |
|-----------|------|--------|
| Hierarchical CSSR | `cssr_discovery/hierarchical_cssr.py` | 🔴 Not started |
| ε-Machine export | `cssr_discovery/epsilon_machine.py` | 🔴 Not started |
| Auto parameter selection | `cssr_discovery/auto_params.py` | 🔴 Not started |
| Neural map projection | `cssr_discovery/neural_projection.py` | 🔴 Not started |
| Witness analysis | `analyze_sequence_overlap.py` | ✅ Complete |
| Linear probes | `probe_machine_sync.py` | ✅ Complete |

### 5.3 Dependencies

```toml
# pyproject.toml additions
[project.dependencies]
scikit-learn = ">=1.3"      # Clustering, probes
scipy = ">=1.11"            # JS divergence
networkx = ">=3.0"          # ε-machine graphs
graphviz = ">=0.20"         # Visualization
```

---

## 6. Success Metrics

### 6.1 Quantitative

| Metric | Target | Current Best |
|--------|--------|--------------|
| Env-View ARI (Grid World) | 1.00 | 1.00 ✅ |
| Agent-View ARI (Grid World) | > 0.80 | 0.00 ❌ |
| Agent-View ARI (Simple) | > 0.90 | 0.77 |
| State count accuracy | ±1 of ground truth | ±2 (gm_seven_union) |
| Loss gap (ε-machine vs neural) | < 0.05 nats | 0.11 nats |

### 6.2 Qualitative

- [ ] Recovered ε-machines are human-interpretable
- [ ] Factorization enables meaningful transfer (physics → new agent)
- [ ] Method scales to environments with > 100 states
- [ ] Algorithm runs in < 1 hour for moderate datasets (100k tokens)

---

## 7. Timeline

| Phase | Duration | Start | End |
|-------|----------|-------|-----|
| Phase 1: Hierarchical Agent | 2 weeks | Week 1 | Week 2 |
| Phase 2: Auto Parameters | 1 week | Week 3 | Week 3 |
| Phase 3: ε-Machine Export | 1 week | Week 4 | Week 4 |
| Phase 4: Neural Projection | 1 week | Week 5 | Week 5 |
| Phase 5: Scaling | 2 weeks | Week 6 | Week 7 |
| Documentation & Paper | 1 week | Week 8 | Week 8 |

---

## 8. Open Questions

1. **Theoretical**: Can we prove that the Env-View equivalence relation recovers the true environment ε-machine under mild assumptions on the agent's policy?

2. **Algorithmic**: What is the optimal way to align intent modes across states in hierarchical agent factorization? (Graph matching? Spectral methods?)

3. **Practical**: How sensitive is the method to model quality? (Undertrained models, miscalibrated probabilities)

4. **Scaling**: Can we use approximate nearest-neighbor methods to avoid $O(N^2)$ pairwise JS computations?

---

## 9. References

- [CSSR Algorithm (Shalizi & Shalizi, 2002)](https://arxiv.org/abs/cs/0210025)
- [Epsilon Machines and Computational Mechanics (Crutchfield & Young, 1989)](https://doi.org/10.1103/PhysRevLett.63.105)
- [World Models (Ha & Schmidhuber, 2018)](https://arxiv.org/abs/1803.10122)
- [Empowerment as Intrinsic Motivation (Klyubin et al., 2005)](https://doi.org/10.1007/11553090_12)

---

## 10. Parameter Predictions: Expected $(L, k)$ Values

Based on the structure of each environment and the lessons from the mixed-machine experiments, here are predictions for what $(L, k)$ values should work well.

### 10.1 General Principles

From the gm_seven_union experiments:
- **$L$ matters more than $k$**: Information is in history length, not rollout horizon
- **$k$ resolves state aliasing**: Use $k=2$ or $k=3$ when states have similar 1-step emissions
- **$L$ should exceed sync length**: The history must contain enough structure to uniquely identify the state

### 10.2 Q2: Agency Detection (Controlled vs Random)

**Environment Structure:**
- 4 states: `{A1, A2, B1, B2}`
- Observations: 4 tokens
- Actions: 2 tokens (`flip`, `hold`)
- Region switch: 5% per step

**Env View Prediction:**
| Parameter | Value | Reasoning |
|-----------|-------|-----------|
| $L$ | **4** | Need 2 obs-act pairs. Single pair is ambiguous (A1 vs B1 both emit distinct obs). Two pairs reveal dynamics. |
| $k$ | **1** | States have distinct observation distributions under intervention. No aliasing expected. |

**Agent View Prediction:**
- Not applicable (agent is reactive, no latent intent states)

---

### 10.3 Q3: Simple Factorization (Swap Machine + 2-Mode Agent)

**Environment Structure:**
- 2 physical states: `{L, R}`
- Observations: 2 tokens
- Actions: 2 tokens (`left`, `right`)
- Deterministic dynamics

**Env View Prediction:**
| Parameter | Value | Reasoning |
|-----------|-------|-----------|
| $L$ | **2** | Single obs-act pair sufficient. `obs_L` uniquely identifies state L. |
| $k$ | **1** | Perfect separation: $P(O_{next} \mid L, a) \neq P(O_{next} \mid R, a)$ for all $a$. |

**Agent View Prediction:**
| Parameter | Value | Reasoning |
|-----------|-------|-----------|
| $L$ | **4–6** | Mode switches at 1%. Need ~3 action samples to reliably distinguish $P(left)=0.9$ vs $P(left)=0.1$. |
| $k$ | **1** | Intent is in action distribution, not observation dynamics. |

---

### 10.4 Q4: Grid World (9-State + Goal-Directed Agent)

**Environment Structure:**
- 9 physical states (3×3 grid)
- Observations: 9 tokens
- Actions: 5 tokens (`up`, `down`, `left`, `right`, `stay`)
- Deterministic dynamics with wall bounces

**Env View Prediction:**
| Parameter | Value | Reasoning |
|-----------|-------|-----------|
| $L$ | **2** | Single obs uniquely identifies location. $L=2$ (one obs-act pair) is sufficient. |
| $k$ | **1** | Each state has unique affordance signature (which actions cause which transitions). |

**Agent View (Global) Prediction:**
- **Will fail** regardless of $(L, k)$
- Reason: Policy is state-dependent. "Seek_0" at position 5 means "go left", but at position 1 means "go up". No global action distribution signature exists.

**Agent View (Hierarchical) Prediction:**
| Parameter | Value | Reasoning |
|-----------|-------|-----------|
| $L$ | **4–8** | Need enough history to see multiple actions from same goal mode. Goal switches at 0.5%. |
| $k$ | **1** | After conditioning on physical state, intent is in action distribution. |

---

### 10.5 Mixed Machines (gm_seven_union)

**Environment Structure:**
- 9 ground-truth states across 2 machines
- 2 tokens (`0`, `1`)
- No actions (pure observation process)

**Findings from Experiments:**
| Metric | Value | Source |
|--------|-------|--------|
| Combinatorial sync length | $L \approx 29$ | Substring overlap analysis |
| Statistical sync length | $L \approx 17$ | Linear probe achieves 100% accuracy |
| Practical sync length | $L \approx 6$ | CSSR with $k=3$ finds 11 states (close to 9) |

**Prediction:**
| Parameter | Value | Reasoning |
|-----------|-------|-----------|
| $L$ | **6–8** | Captures enough context for near-deterministic belief. |
| $k$ | **2–3** | Needed to resolve state aliasing (e.g., GM:A vs Human:baa have JS ≈ 0.03 bits at $k=2$). |
| $\epsilon$ | **0.02–0.03 bits** | Tight enough to separate, loose enough to avoid over-split. |

---

### 10.6 Future Environments (Predictions)

| Environment | States | $L$ Prediction | $k$ Prediction | Notes |
|-------------|--------|----------------|----------------|-------|
| **MiniGrid 6×6** | 36 | 4–6 | 1–2 | Partial observability increases $L$. Agent sees local view, need multiple steps to localize. |
| **Atari Pong** | ∞ | 8–16 | 1 | Position/velocity encoded in frames. Need short history for motion. |
| **TextWorld** | Variable | 10–20 | 1 | Text commands are long. State depends on object inventory + location. |

---

### 10.7 Summary Table

| Experiment | Env View $(L, k)$ | Agent View $(L, k)$ | Notes |
|------------|-------------------|---------------------|-------|
| Q2: Agency | $(4, 1)$ | N/A | Simple |
| Q3: Simple | $(2, 1)$ | $(4\text{–}6, 1)$ | Easy factorization |
| Q4: Grid | $(2, 1)$ | Hierarchical: $(4\text{–}8, 1)$ | Agent view requires hierarchy |
| Mixed (gm7) | $(6\text{–}8, 2\text{–}3)$ | N/A | Non-Markovian, approximate |

**Rule of Thumb:**
- $L \approx 2 \times \text{(tokens per state cycle)}$ for deterministic environments
- $L \approx \text{sync length} + 2$ for stochastic environments
- $k = 1$ unless states have similar 1-step emissions, then $k = 2\text{–}3$
