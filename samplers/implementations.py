import numpy as np
from scipy.stats import qmc
from .base import BaseSampler
from distributions.base import Distribution

class TransformingSampler(BaseSampler):
    """Base class for samplers that use inverse transform sampling."""
    
    def transform_samples(self, uniform_samples: np.ndarray) -> np.ndarray:
        """Transform uniform samples to target distribution."""
        return self.distribution.inverse_cdf(uniform_samples)

class UnitCubeSampler(TransformingSampler):
    """Uniform random sampling with inverse transform."""
    
    def generate_samples(self, n_samples: int) -> np.ndarray:
        uniform_samples = np.random.uniform(
            low=0, high=1, 
            size=(n_samples, self.n_dimensions)
        )
        return self.transform_samples(uniform_samples)

class SobolSampler(TransformingSampler):
    """Quasi-Monte Carlo sampler using Sobol sequences with inverse transform."""
    
    def __init__(self, scramble: bool = True):
        super().__init__()
        self.scramble = scramble
        self.sampler = None
    
    def setup(self, distribution: Distribution, n_dimensions: int):
        super().setup(distribution, n_dimensions)
        self.sampler = qmc.Sobol(
            d=n_dimensions, 
            scramble=self.scramble
        )
    
    def generate_samples(self, n_samples: int) -> np.ndarray:
        uniform_samples = self.sampler.random_base2(
            m=int(np.ceil(np.log2(n_samples)))
        )
        return self.transform_samples(uniform_samples)

class MetropolisHastingsSampler(BaseSampler):
    """Metropolis-Hastings MCMC sampler."""
    
    def __init__(self, step_size: float = 0.1, burn_in: int = 1000):
        super().__init__()
        self.step_size = step_size
        self.burn_in = burn_in
    
    def generate_samples(self, n_samples: int) -> np.ndarray:
        samples = np.zeros((n_samples + self.burn_in, self.n_dimensions))
        current = np.random.randn(self.n_dimensions)  # Start from random point
        
        for i in range(n_samples + self.burn_in):
            # Propose new sample
            proposal = current + np.random.randn(self.n_dimensions) * self.step_size
            
            # Compute acceptance ratio
            log_ratio = (
                self.distribution.log_pdf(proposal) - 
                self.distribution.log_pdf(current)
            )
            
            # Accept or reject
            if np.log(np.random.random()) < log_ratio:
                current = proposal
                
            samples[i] = current
            
        return samples[self.burn_in:] 