from abc import ABC, abstractmethod
import numpy as np
from distributions.base import Distribution

class BaseSampler(ABC):
    """Base class for sampling strategies."""
    
    def __init__(self):
        self.distribution = None
        self.n_dimensions = None
    
    def setup(self, distribution: Distribution, n_dimensions: int):
        """Initialize the sampler with a distribution and dimensionality."""
        self.distribution = distribution
        self.n_dimensions = n_dimensions
    
    @abstractmethod
    def generate_samples(self, n_samples: int) -> np.ndarray:
        """Generate samples from the distribution.
        
        Args:
            n_samples: Number of samples to generate
            
        Returns:
            Array of shape (n_samples, n_dimensions) containing the samples
        """
        pass 