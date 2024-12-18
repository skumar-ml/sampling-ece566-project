import numpy as np
from scipy.stats import qmc
from .base import BaseSampler
from distributions.base import Distribution
from distributions.implementations import GMMDistribution

class TransformingSampler(BaseSampler):
    """Base class for samplers that use inverse transform sampling."""
    
    def transform_samples(self, uniform_samples: np.ndarray) -> np.ndarray:
        """Transform uniform samples to target distribution."""
        return self.distribution.inverse_cdf(uniform_samples)

class MonteCarloSampler(BaseSampler):
    """Basic Monte Carlo sampler that directly samples from the distribution."""
    
    def generate_samples(self, n_samples: int) -> np.ndarray:
        """Generate samples directly from the distribution."""
        if self.distribution is None:
            raise RuntimeError("Must call setup() before generating samples")
        return self.distribution.sample(n_samples)

class SobolSampler(TransformingSampler):
    """Quasi-Monte Carlo sampler using Sobol sequences with inverse transform."""
    
    def __init__(self, scramble: bool = True):
        super().__init__()
        self.scramble = scramble
        self.sampler = None
    
    def setup(self, distribution: Distribution, n_dimensions: int):
        """
        Setup the sampler. For GMM, adds an extra dimension for component selection.
        """
        super().setup(distribution, n_dimensions)
        
        # Add extra dimension if distribution is GMM
        actual_dims = n_dimensions
        if isinstance(distribution, GMMDistribution):
            actual_dims += 1  # Extra dimension for component selection
            
        self.sampler = qmc.Sobol(
            d=actual_dims, 
            scramble=self.scramble
        )
    
    def generate_samples(self, n_samples: int) -> np.ndarray:
        """Generate Sobol sequence samples."""
        if self.sampler is None:
            raise RuntimeError("Must call setup() before generating samples")
            
        # Generate samples using power of 2
        m = int(np.ceil(np.log2(n_samples)))
        u = self.sampler.random_base2(m=m)
        
        # Take only the requested number of samples
        u = u[:n_samples]
        
        return self.transform_samples(u)

class TruncatedMHSampler(BaseSampler):
    """
    Metropolis-Hastings sampler for truncated distributions:
    p*(x) ∝ 1(x > thresholds) * target_dist.pdf(x)
    """
    def __init__(self, thresholds: np.ndarray, step_size: float = 0.1, burn_in: int = 1000, log_target_density=None):
        super().__init__()
        self.thresholds = thresholds
        self.step_size = step_size
        self.burn_in = burn_in
        self.custom_log_target_density = log_target_density
        
    def setup(self, distribution: Distribution, n_dimensions: int):
        self.distribution = distribution
        self.n_dimensions = n_dimensions
        
        # Initialize starting point above thresholds
        self.current_state = self.generate_valid_initial_state()
    
    def log_target_density(self, x):
        if self.custom_log_target_density is not None:
            return self.custom_log_target_density(x)
        return self.distribution.log_pdf(x)
    
    def generate_valid_initial_state(self):
        while True:
            x = self.distribution.mean + np.random.randn(self.n_dimensions)
            if np.all(x > self.thresholds):
                return x
    
    def propose_next_state(self):
        return self.current_state + self.step_size * np.random.randn(self.n_dimensions)
    
    def generate_samples(self, n_samples: int) -> np.ndarray:
        samples = np.zeros((n_samples + self.burn_in, self.n_dimensions))
        samples[0] = self.current_state
        
        for i in range(1, n_samples + self.burn_in):
            proposed_state = self.propose_next_state()
            
            current_log_density = self.log_target_density(self.current_state)
            proposed_log_density = self.log_target_density(proposed_state)
            
            log_acceptance_ratio = proposed_log_density - current_log_density
            
            if np.log(np.random.rand()) < log_acceptance_ratio:
                self.current_state = proposed_state
            
            samples[i] = self.current_state
        
        return samples[self.burn_in:]