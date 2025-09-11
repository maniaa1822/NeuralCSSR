# FER Hypothesis Testing with NanoGPT and Epsilon Machines: Research Plan

## Research Question

**Core Hypothesis**: SGD-trained neural networks develop Fractured Entangled Representations (FER) that are inferior to representations learned through evolutionary or multi-task training approaches.

**Test Environment**: Use epsilon machines as ground truth for optimal sequential representations, comparing different training paradigms on causal state recovery tasks.

## Phase 1: Baseline Implementation & Validation

### 1.1 Ground Truth Causal State Recovery

```python
def recover_golden_mean_causal_states(sequences):
    """Compute ground truth causal states for Golden Mean process"""
    causal_states = []
    
    for sequence in sequences:
        sequence_states = []
        for pos, symbol in enumerate(sequence):
            if pos == 0:
                state = 'A' if symbol == '0' else 'B'
            else:
                # State depends on previous symbol
                prev_symbol = sequence[pos-1]
                state = 'A' if prev_symbol == '0' else 'B'
            sequence_states.append(state)
        causal_states.append(sequence_states)
    
    return causal_states
```

### 1.2 Basic Linear Probe Test

```python
def test_causal_state_decodability(model, sequences, causal_states):
    """Test if causal states are linearly decodable after training"""
    
    hidden_states = model.get_hidden_states(sequences)
    
    X_train, X_test, y_train, y_test = train_test_split(
        hidden_states, causal_states, test_size=0.2
    )
    
    probe = LogisticRegression()
    probe.fit(X_train, y_train)
    accuracy = probe.score(X_test, y_test)
    
    return {
        'accuracy': accuracy,
        'chance_level': 1.0 / len(np.unique(causal_states)),
        'probe_weights': probe.coef_
    }
```

### 1.3 Validation Metrics

- **Causal State Classification Accuracy**: Can linear probe recover causal states?
- **Transition Constraint Validation**: Do recovered states respect epsilon machine rules?
- **Representation Geometry**: Clustering coherence, separation ratios

## Phase 2: Core FER Comparison

### 2.1 Training Paradigms to Compare

1. **Standard SGD (Baseline)**
   - Single-task next-token prediction
   - Standard hyperparameters
   - Expected: Fractured representations

2. **Multi-Task SGD**
   - Simultaneous training on multiple epsilon machine tasks
   - Expected: Better than baseline, but potentially still fractured

3. **Evolutionary Training**
   - Population-based optimization
   - Fitness based on causal state recovery
   - Expected: Unified representations (anti-FER)

4. **Meta-Learning Curriculum**
   - Learn to select optimal training examples
   - Adaptive example selection based on representation quality
   - Expected: Adaptive avoidance of fragmentation

### 2.2 Multi-Task Training Design

```python
def multi_task_epsilon_training(model, epsilon_machine, sequences):
    """DINO-inspired multi-task training for unified representations"""
    
    tasks = {
        'next_token': next_token_prediction_loss,
        'current_state': causal_state_classification_loss,
        'future_state': predict_state_n_steps_ahead_loss,
        'anomaly_detection': sequence_anomaly_detection_loss,
        'transition_validity': valid_transition_classification_loss,
        'sequence_completion': masked_sequence_modeling_loss
    }
    
    for batch in dataloader:
        total_loss = sum(task_fn(model, batch, epsilon_machine) 
                        for task_fn in tasks.values())
        total_loss.backward()
        optimizer.step()
```

## Phase 3: DINO-Inspired Modifications for Unified Representations

### 3.1 Scale and Architecture

- **Larger Models**: 24 layers vs 12, 1024 embedding vs 512
- **Longer Training**: 1000 epochs vs 100, 10k sequences per epoch
- **Higher Capacity**: Test if scale naturally overcomes FER

### 3.2 Curriculum Learning

```python
def developmental_curriculum(model, epsilon_machine):
    """Hierarchical training inspired by DINO developmental trajectory"""
    
    # Stage 1: Simple pattern recognition
    train_on_tasks(model, ['next_symbol', 'symbol_frequency'])
    
    # Stage 2: Local structure  
    train_on_tasks(model, ['local_patterns', 'short_dependencies'])
    
    # Stage 3: Global causal structure
    train_on_tasks(model, ['causal_states', 'long_range_prediction'])
```

### 3.3 Cross-Machine Training

```python
def multi_machine_training(model):
    """Train on multiple epsilon machines for abstract causal reasoning"""
    
    machines = [
        EvenProcess(),      # 2 states
        GoldenMean(),       # 3 states  
        RIP(),              # 4 states
        RandomDOT(k=4),     # Complex structure
    ]
    
    # Mix sequences from all machines
    mixed_sequences = []
    for machine in machines:
        mixed_sequences.extend(machine.generate_sequences(1000))
    
    return train_multi_task(model, mixed_sequences, machines)
```

### 3.4 Self-Supervised Objectives

- **Masked Sequence Modeling**: Predict masked symbols requiring causal understanding
- **Contrastive Learning**: Learn that causally equivalent histories should be similar
- **Sequence Ordering**: Predict if sequence transitions are valid

## Phase 4: Advanced Analysis Methods

### 4.1 Representational Quality Metrics

```python
def comprehensive_representation_analysis(model, epsilon_machine):
    """Multi-perspective analysis of representation quality"""
    
    return {
        'causal_state_alignment': measure_state_classification_accuracy(),
        'geometric_coherence': measure_clustering_purity(),
        'redundancy_analysis': measure_representational_redundancy(),
        'transfer_learning': test_cross_machine_transfer(),
        'developmental_trajectory': measure_half_times()
    }
```

### 4.2 Developmental Trajectory Analysis

Track when different aspects of causal structure emerge during training:

```python
def measure_representation_half_times(model, checkpoints, epsilon_machine):
    """When do different causal features become learnable?"""
    
    half_times = {}
    for feature in ['states', 'transitions', 'predictions']:
        alignment_scores = []
        for checkpoint in checkpoints:
            score = measure_feature_alignment(checkpoint, feature)
            alignment_scores.append(score)
        
        final_score = alignment_scores[-1]
        half_time = find_half_time(alignment_scores, final_score * 0.5)
        half_times[feature] = half_time
    
    return half_times
```

### 4.3 Geometric Analysis

- **Participation Ratio**: Measure embedding dimensionality
- **Linear Separability**: Test if causal states are linearly separable
- **Manifold Learning**: Analyze representational geometry using t-SNE, UMAP

## Phase 5: Hypothesis Testing

### 5.1 Key Predictions

**If FER hypothesis is correct:**
- SGD models: Low causal state recovery, high redundancy, poor transfer
- Evolved models: High causal state recovery, low redundancy, good transfer
- Multi-task models: Intermediate performance
- Scale alone insufficient to overcome FER

**If FER hypothesis is wrong:**
- All training methods converge to similar representation quality
- Scale consistently improves all methods
- Task structure matters more than optimization procedure

### 5.2 Statistical Testing

```python
def statistical_comparison(sgd_results, evolved_results, multitask_results):
    """Statistical significance testing of representation quality differences"""
    
    metrics = ['accuracy', 'coherence', 'redundancy', 'transfer']
    comparisons = {}
    
    for metric in metrics:
        # Paired t-tests between methods
        sgd_vs_evolved = ttest_rel(sgd_results[metric], evolved_results[metric])
        sgd_vs_multitask = ttest_rel(sgd_results[metric], multitask_results[metric])
        
        comparisons[metric] = {
            'sgd_vs_evolved': sgd_vs_evolved,
            'sgd_vs_multitask': sgd_vs_multitask
        }
    
    return comparisons
```

## Phase 6: Extensions and Applications

### 6.1 Meta-Learning Integration

- **Learned Curriculum**: Meta-learn optimal example selection
- **Representation Quality Prediction**: Learn to predict which examples improve representations
- **Adaptive Training**: Real-time adjustment based on representation metrics

### 6.2 Scaling Analysis

- **Model Size Effects**: Systematic analysis across model sizes
- **Data Efficiency**: How much data needed for unified representations?
- **Computational Trade-offs**: Cost of unified vs fractured representations

## Expected Timeline and Outcomes

**Months 1-2**: Baseline implementation and validation
**Months 3-4**: Core FER comparison experiments
**Months 5-6**: DINO-inspired modifications and analysis
**Months 7-8**: Advanced analysis and statistical testing

**Key Deliverables:**
1. Empirical test of FER hypothesis with ground truth validation
2. Novel insights into representation quality in sequence models
3. Practical training methods for improved causal reasoning
4. Theoretical connections between optimization and representation geometry

## Implementation Priority

1. **Start with Golden Mean machine** (simple, well-understood)
2. **Implement basic linear probe pipeline** (validate ground truth recovery)
3. **Compare SGD vs multi-task training** (core FER test)
4. **Add evolutionary training** (strongest FER prediction)
5. **Scale up and analyze** (comprehensive evaluation)

This research plan bridges theoretical questions about optimization and representation learning with practical advances in training neural networks for sequential reasoning tasks.