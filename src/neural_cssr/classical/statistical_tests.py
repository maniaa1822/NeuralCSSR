"""Statistical tests for CSSR sufficiency testing."""

import numpy as np
from scipy import stats
from typing import Dict, List, Tuple, Optional
from collections import defaultdict, Counter
import math


class StatisticalTests:
    """Statistical tests for determining if distributions are significantly different."""
    
    def __init__(self, significance_level: float = 0.05):
        self.significance_level = significance_level
        self.kl_threshold = 0.1  # Default KL divergence threshold
    
    def chi_square_test(
        self, 
        dist1: Dict[str, int], 
        dist2: Dict[str, int],
        min_count: int = 5
    ) -> Tuple[bool, float, float]:
        """Chi-square test for comparing two empirical distributions.
        
        Returns:
            (is_significantly_different, chi2_statistic, p_value)
        """
        # Get all symbols that appear in either distribution
        all_symbols = set(dist1.keys()) | set(dist2.keys())
        
        if len(all_symbols) < 2:
            return False, 0.0, 1.0
        
        # Build contingency table
        observed1 = []
        observed2 = []
        
        for symbol in sorted(all_symbols):
            count1 = dist1.get(symbol, 0)
            count2 = dist2.get(symbol, 0)
            observed1.append(count1)
            observed2.append(count2)
        
        # Convert to numpy arrays
        observed = np.array([observed1, observed2])
        
        # Check if we have enough data
        total1 = sum(observed1)
        total2 = sum(observed2)
        
        if total1 < min_count or total2 < min_count:
            return False, 0.0, 1.0
        
        # Perform chi-square test
        try:
            chi2, p_value, dof, expected = stats.chi2_contingency(observed)
            
            # Check if expected frequencies are large enough
            if np.any(expected < 1):
                return False, 0.0, 1.0
            
            is_different = p_value < self.significance_level
            return is_different, chi2, p_value
            
        except (ValueError, ZeroDivisionError):
            return False, 0.0, 1.0
    
    def kl_divergence_test(
        self,
        dist1: Dict[str, int],
        dist2: Dict[str, int],
        threshold: float = 0.1
    ) -> Tuple[bool, float]:
        """KL divergence test for comparing distributions.
        
        Returns:
            (is_significantly_different, kl_divergence)
        """
        # Get all symbols
        all_symbols = set(dist1.keys()) | set(dist2.keys())
        
        if len(all_symbols) < 2:
            return False, 0.0
        
        # Convert to probability distributions with smoothing
        total1 = sum(dist1.values())
        total2 = sum(dist2.values())
        
        if total1 == 0 or total2 == 0:
            return False, 0.0
        
        # Add-one smoothing to avoid zero probabilities
        smooth_factor = 1e-10
        
        p = []
        q = []
        
        for symbol in sorted(all_symbols):
            count1 = dist1.get(symbol, 0)
            count2 = dist2.get(symbol, 0)
            
            prob1 = (count1 + smooth_factor) / (total1 + len(all_symbols) * smooth_factor)
            prob2 = (count2 + smooth_factor) / (total2 + len(all_symbols) * smooth_factor)
            
            p.append(prob1)
            q.append(prob2)
        
        # Calculate KL divergence
        kl_div = 0.0
        for p_i, q_i in zip(p, q):
            if p_i > 0:
                kl_div += p_i * math.log(p_i / q_i)
        
        is_different = kl_div > threshold
        return is_different, kl_div
    
    def permutation_test(
        self,
        dist1: Dict[str, int],
        dist2: Dict[str, int],
        n_permutations: int = 1000
    ) -> Tuple[bool, float]:
        """Permutation test for comparing distributions.
        
        Returns:
            (is_significantly_different, p_value)
        """
        # Get all symbols and create combined dataset
        all_symbols = set(dist1.keys()) | set(dist2.keys())
        
        if len(all_symbols) < 2:
            return False, 1.0
        
        # Create combined dataset
        data1 = []
        data2 = []
        
        for symbol in all_symbols:
            count1 = dist1.get(symbol, 0)
            count2 = dist2.get(symbol, 0)
            data1.extend([symbol] * count1)
            data2.extend([symbol] * count2)
        
        if len(data1) == 0 or len(data2) == 0:
            return False, 1.0
        
        combined_data = data1 + data2
        n1, n2 = len(data1), len(data2)
        
        # Calculate observed test statistic (chi-square)
        def calculate_chi_square(group1, group2):
            counter1 = Counter(group1)
            counter2 = Counter(group2)
            
            observed1 = []
            observed2 = []
            
            for symbol in sorted(all_symbols):
                observed1.append(counter1.get(symbol, 0))
                observed2.append(counter2.get(symbol, 0))
            
            observed = np.array([observed1, observed2])
            
            try:
                chi2, _, _, _ = stats.chi2_contingency(observed)
                return chi2
            except (ValueError, ZeroDivisionError):
                return 0.0
        
        observed_stat = calculate_chi_square(data1, data2)
        
        # Perform permutations
        more_extreme = 0
        
        for _ in range(n_permutations):
            # Shuffle combined data
            np.random.shuffle(combined_data)
            
            # Split into two groups
            perm_group1 = combined_data[:n1]
            perm_group2 = combined_data[n1:]
            
            # Calculate test statistic
            perm_stat = calculate_chi_square(perm_group1, perm_group2)
            
            if perm_stat >= observed_stat:
                more_extreme += 1
        
        p_value = more_extreme / n_permutations
        is_different = p_value < self.significance_level
        
        return is_different, p_value
    
    def sufficient_statistics_test(
        self,
        dist1: Dict[str, int],
        dist2: Dict[str, int],
        test_type: str = "chi_square",
        **kwargs
    ) -> Tuple[bool, Dict]:
        """Test if two distributions are statistically indistinguishable.
        
        Args:
            dist1, dist2: Empirical distributions as symbol -> count dicts
            test_type: "chi_square", "kl_divergence", or "permutation"
            
        Returns:
            (are_different, test_results)
        """
        if test_type == "chi_square":
            is_different, statistic, p_value = self.chi_square_test(dist1, dist2, **kwargs)
            return is_different, {
                'test': 'chi_square',
                'statistic': statistic,
                'p_value': p_value,
                'significant': is_different
            }
        
        elif test_type == "kl_divergence":
            # Use the configured threshold if not provided in kwargs
            if 'threshold' not in kwargs:
                kwargs['threshold'] = self.kl_threshold
            is_different, kl_div = self.kl_divergence_test(dist1, dist2, **kwargs)
            return is_different, {
                'test': 'kl_divergence',
                'kl_divergence': kl_div,
                'threshold': kwargs['threshold'],
                'significant': is_different
            }
        
        elif test_type == "permutation":
            is_different, p_value = self.permutation_test(dist1, dist2, **kwargs)
            return is_different, {
                'test': 'permutation',
                'p_value': p_value,
                'significant': is_different
            }
        
        else:
            raise ValueError(f"Unknown test type: {test_type}")
    
    def multiple_test_correction(self, p_values: List[float], method: str = "bonferroni") -> List[float]:
        """Apply multiple testing correction."""
        if method == "bonferroni":
            return [min(1.0, p * len(p_values)) for p in p_values]
        elif method == "holm":
            sorted_indices = sorted(range(len(p_values)), key=lambda i: p_values[i])
            corrected = [0.0] * len(p_values)
            
            for rank, idx in enumerate(sorted_indices):
                correction_factor = len(p_values) - rank
                corrected[idx] = min(1.0, p_values[idx] * correction_factor)
            
            return corrected
        else:
            raise ValueError(f"Unknown correction method: {method}")