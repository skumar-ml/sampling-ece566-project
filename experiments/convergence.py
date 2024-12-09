import numpy as np
from typing import Dict, Any
from dataclasses import dataclass

@dataclass
class ConvergenceData:
    """Container for convergence analysis results."""
    running_means: np.ndarray
    running_variances: np.ndarray
    sample_indices: np.ndarray

class ConvergenceAnalysis:
    """Analyzes convergence of sampling estimates."""
    
    def analyze(self, values: np.ndarray) -> ConvergenceData:
        """
        Analyze convergence of sampling estimates.
        
        Args:
            values: Array of function values from samples
            
        Returns:
            ConvergenceData containing running means and variances
        """
        n_samples = len(values)
        running_means = np.zeros(n_samples)
        running_variances = np.zeros(n_samples)
        
        # Compute running statistics
        for i in range(n_samples):
            running_means[i] = np.mean(values[:i+1])
            running_variances[i] = np.var(values[:i+1]) if i > 0 else 0
            
        return ConvergenceData(
            running_means=running_means,
            running_variances=running_variances,
            sample_indices=np.arange(1, n_samples + 1)
        ) 