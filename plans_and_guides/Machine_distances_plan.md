Machine Distance Metrics Implementation Plan
Overview
Implement three core distance metrics to quantitatively compare CSSR-discovered machines against ground truth machines from the unified dataset generation pipeline.
Implementation Requirements
Input Data Structures
python# Discovered machine from CSSR results
discovered_machine = {
    "states": {
        "State_0": {
            "symbol_distribution": {"0": 0.522, "1": 0.478},
            "entropy": 0.998,
            "observations": 38805
        },
        "State_1": {
            "symbol_distribution": {"0": 0.755, "1": 0.245}, 
            "entropy": 0.803,
            "observations": 6472
        }
        # ... more states
    },
    "num_states": 5,
    "transitions": {},  # Will be populated if available
    "statistical_complexity": 0.56,  # From CSSR if available
    "entropy_rate": 0.96  # From CSSR if available
}

# Ground truth machines from dataset metadata
ground_truth_machines = [
    {
        "machine_id": "1",
        "states": {
            "S0": {"distribution": {"0": 0.5, "1": 0.5}},
            "S1": {"distribution": {"0": 0.5, "1": 0.5}}
        },
        "properties": {
            "num_states": 2,
            "statistical_complexity": 1.0,
            "entropy_rate": 1.0,
            "is_topological": True
        },
        "samples_count": 5000,
        "weight": 1.0
    },
    {
        "machine_id": "10", 
        "states": {
            "S0": {"distribution": {"0": 0.8, "1": 0.2}},
            "S1": {"distribution": {"0": 0.3, "1": 0.7}},
            "S2": {"distribution": {"0": 0.5, "1": 0.5}}
        },
        "properties": {
            "num_states": 3,
            "statistical_complexity": 1.585,
            "entropy_rate": 0.868,
            "is_topological": False
        },
        "samples_count": 1500,
        "weight": 1.5
    }
]
Metric 1: State Mapping Distance
Specification
Find optimal assignment of discovered states to ground truth states using Hungarian algorithm, based on symbol distribution similarity.
Implementation Details
pythonclass StateMappingDistance:
    def compute(self, discovered_machine, ground_truth_machines):
        """
        Returns:
        {
            "optimal_assignment": [...],
            "total_cost": float,
            "per_state_costs": [...],
            "unmatched_discovered_states": [...],
            "unmatched_true_states": [...]
        }
        """
Algorithm Steps

Create cost matrix: For each (discovered_state, true_state) pair, compute Jensen-Shannon divergence between symbol distributions
Handle size mismatch: Pad smaller matrix with high-cost dummy states
Apply Hungarian algorithm: Find minimum-cost bipartite matching
Extract metrics: Total cost, per-state costs, unmatched states

Dependencies

scipy.optimize.linear_sum_assignment for Hungarian algorithm
scipy.spatial.distance.jensenshannon for JS divergence

Metric 2: Transition Structure Distance
Specification
Compare transition graph connectivity patterns using graph edit distance or spectral methods.
Implementation Details
pythonclass TransitionStructureDistance:
    def compute(self, discovered_machine, ground_truth_machines):
        """
        Returns:
        {
            "graph_edit_distance": float,
            "spectral_distance": float, 
            "connectivity_similarity": float,
            "transition_coverage": float
        }
        """
Algorithm Steps

Build transition graphs: Convert state machines to NetworkX directed graphs
Merge ground truth graphs: Combine multiple machines into unified graph with proper state naming
Compute graph edit distance: Use NetworkX graph edit distance
Compute spectral distance: Compare graph Laplacian eigenvalues
Analyze connectivity: Compare in-degree, out-degree distributions

Dependencies

networkx for graph operations and edit distance
numpy for spectral analysis

Note
If transition information not available in CSSR output, implement fallback using only state connectivity patterns.
Metric 3: Symbol Distribution Distance
Specification
Compare state-wise symbol emission distributions, finding optimal matching between discovered and true states.
Implementation Details
pythonclass SymbolDistributionDistance:
    def compute(self, discovered_machine, ground_truth_machines):
        """
        Returns:
        {
            "average_js_divergence": float,
            "max_js_divergence": float,
            "min_js_divergence": float,
            "state_mappings": [
                {
                    "discovered_state": "State_0",
                    "best_match": {"machine_id": "10", "state_id": "S0"},
                    "js_divergence": 0.023,
                    "distance_rank": 1
                }
            ],
            "coverage_score": float  # How many true states have good matches
        }
        """
Algorithm Steps

Flatten ground truth states: Create single list of all states across machines with proper machine_id tracking
For each discovered state:

Compute JS divergence to all ground truth states
Find best match (minimum divergence)
Record distance and mapping


Compute aggregate metrics:

Average, max, min divergences
Coverage score (fraction of true states well-matched)


Bidirectional analysis: Also find best discovered match for each true state

Distance Function
Use Jensen-Shannon divergence for symmetric, bounded distance between probability distributions.
Integration Class
Main Interface
pythonclass MachineDistanceCalculator:
    def __init__(self):
        self.state_mapping = StateMappingDistance()
        self.transition_structure = TransitionStructureDistance()
        self.symbol_distribution = SymbolDistributionDistance()
    
    def compute_all_distances(self, discovered_machine, ground_truth_machines):
        """
        Compute all three distance metrics.
        
        Returns:
        {
            "state_mapping_distance": {...},
            "transition_structure_distance": {...}, 
            "symbol_distribution_distance": {...},
            "summary": {
                "overall_distance_score": float,
                "best_metric": str,
                "confidence": float
            }
        }
        """
    
    def generate_report(self, results, output_path):
        """Generate comprehensive markdown report with visualizations."""
File Structure
src/evaluation/
├── __init__.py
├── machine_distance.py          # Main MachineDistanceCalculator class
├── metrics/
│   ├── __init__.py
│   ├── state_mapping.py         # StateMappingDistance implementation
│   ├── transition_structure.py  # TransitionStructureDistance implementation
│   └── symbol_distribution.py   # SymbolDistributionDistance implementation
├── utils/
│   ├── __init__.py
│   ├── data_loading.py          # Load CSSR and ground truth data
│   └── visualization.py        # Distance metric visualizations
└── tests/
    ├── test_machine_distance.py
    └── test_data/               # Sample test data
Usage Example
python# Load data
cssr_results = load_cssr_results("datasets/biased_exp/classical_cssr_results.json")
ground_truth = load_ground_truth("datasets/biased_exp/ground_truth/")

# Compute distances
calculator = MachineDistanceCalculator()
distances = calculator.compute_all_distances(cssr_results, ground_truth)

# Generate report
calculator.generate_report(distances, "results/machine_distance_analysis.md")

print(f"Symbol distribution distance: {distances['symbol_distribution_distance']['average_js_divergence']:.4f}")
print(f"State mapping cost: {distances['state_mapping_distance']['total_cost']:.4f}")
Testing Strategy
Unit Tests

Test each metric independently with synthetic data
Verify edge cases (empty machines, single states, perfect matches)
Test mathematical properties (symmetry, triangle inequality where applicable)

Integration Tests

Test on actual CSSR results from biased_exp dataset
Verify results match expected patterns (State_1 should match Machine_10.S0)
Test with multiple ground truth machines

Expected Results Validation
Based on the CSSR results provided:

State_1 (75.5% "0") should closely match Machine_10.S0 (80% "0") - low JS divergence
State_2 (19.2% "0") should closely match Machine_10.S1 (30% "0") - low JS divergence
State_0 should match balanced states from either machine

Dependencies
toml[dependencies]
scipy = ">=1.9.0"     # For Hungarian algorithm and JS divergence
networkx = ">=2.8"    # For graph operations
numpy = ">=1.21.0"    # For numerical computations
matplotlib = ">=3.5"  # For visualizations
Success Criteria

Functional: All three metrics compute without errors on real CSSR data
Sensible: Results align with visual inspection (State_1 ↔ Machine_10.S0 should have low distance)
Robust: Handle edge cases gracefully (unequal state counts, missing transitions)
Interpretable: Clear mapping between discovered and true states with confidence scores