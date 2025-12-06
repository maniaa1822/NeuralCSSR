Walkthrough - Butterfly Machine Implementation
Result: SUCCESS - Full Reconstruction

1. Machine Implementation
Files: 
machines/butterfly_machine.py
, 
machines/
init
.py
.
States: 5 (A, B, C, D, E).
Alphabet: 8 symbols (0-7).
Verified: 
scripts/butterfly_verify.py
 passed state coverage and ground truth checks.
2. Model Training
Config: 
nanoGPT/config/train_butterfly.py
 (Mini model, 512 batch).
Converged: ~1000 steps.
Loss: 0.697 nats (1.006 bits).
Theoretical Limit: 0.6931 nats (1.000 bits).
Status: Model has perfectly learned the machine dynamics.
3. Generalized CSSR Discovery
I implemented 
cssr_discovery/generalized_2_stage.py
 to handle the 8-symbol alphabet (previously binary-only).

Command:

uv run cssr_discovery/generalized_2_stage.py \
    --ckpt nanoGPT/out-butterfly/ckpt.pt \
    --dataset_dir experiments/datasets/butterfly \
    --L 4 --k 1 --tol 0.05
Results: Perfect reconstruction of the ground truth topology:

State 0 (A): ['len1:1']
State 1 (B): ['len1:2', 'len1:4']
State 2 (C): ['len1:6', 'len2:20', 'len2:40']
Note: correctly identifies that '0' is ambiguous unless preceded by '2' or '4'
State 3 (D): ['len1:3', 'len1:5']
State 4 (E): ['len1:7', 'len2:30', 'len2:50']
Note: correctly identifies '0' from '3'/'5'
4. Nucleus Sampling Verification
Added Nucleus Sampling to handle variable branching factors robustly. Command:

uv run cssr_discovery/generalized_2_stage.py \
    ... \
    --nucleus 0.95
Result:

Still perfectly recovers the 5-state machine.
Branching correctly focuses on the 2 equiprobable transitions per state, ignoring negligible probability tails.
5. Benchmarking & Robustness
We tested the impact of Nucleus Sampling on performance and accuracy.

Method	Params	Time	States Recovered	Status
Baseline	TopK=8	~2.92s	5	Accurate
Safe Nucleus	p=0.95	~2.97s	5	Accurate
Aggressive	p=0.45	~2.85s	9	BROKEN
Analysis:

Speed: Negligible difference ($<0.1s$) for this small scale.
Distortion Strategy (Fragmentation):
At p=0.45, the sampler is forced to pick only one of the two valid 50% branches.
Because of random noise in the model's logits (e.g., $P(0)=0.501, P(1)=0.499$), this choice fluctuates across different histories of the same state.
Result: Homogeneous states shatter into multiple pieces.
State A (1) $\to$ shattered into States 0, 8.
State D (3,5) $\to$ shattered into States 3, 5, 6.
State E (7, 30...) $\to$ shattered into States 4, 7.
States B & C survived intact (likely due to consistent bias in the model preferring one transition over the other).
Conclusion
The workflow is fully validated for non-binary alphabets. The model learned the 2-cryptic structure, and the generalized CSSR algorithm successfully extracted the causal states and synchronizing suffixes (Memory Length 2 logic) from the neural network's predictions. Nucleus sampling works correctly when $p$ covers the valid transitions.