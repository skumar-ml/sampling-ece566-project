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
    def __init__(self, thresholds: np.ndarray, step_size: float = 0.3, burn_in: int = 2000):
        super().__init__()
        self.thresholds = thresholds
        self.step_size = step_size  # Reduced for better acceptance rate
        self.burn_in = burn_in  # Increased for better mixing
        self.current_state = None
        self.acceptance_count = 0
        self.total_proposals = 0
    
    def setup(self, distribution: Distribution, n_dimensions: int):
        super().setup(distribution, n_dimensions)
        self.current_state = self._get_initial_state()
    
    def _get_initial_state(self) -> np.ndarray:
        """Initialize chain starting point deterministically."""
        std = np.sqrt(np.diag(self.distribution.cov))
        return self.thresholds + 1.0 * std  # Start slightly above thresholds
    
    def _get_proposal_covariance(self) -> np.ndarray:
        """
        Construct proposal covariance that respects the target's correlation structure
        but allows for adaptive step sizes in each dimension.
        """
        return self.step_size * self.distribution.cov
    
    def _log_target(self, x: np.ndarray) -> float:
        """Compute log of unnormalized target density."""
        if np.any(x <= self.thresholds):
            return -np.inf
        return self.distribution.log_pdf(x)
    
    def generate_samples(self, n_samples: int) -> np.ndarray:
        """Generate samples using M-H algorithm."""
        samples = np.zeros((n_samples + self.burn_in, self.n_dimensions))
        samples[0] = self.current_state
        
        # Get proposal covariance
        proposal_cov = self._get_proposal_covariance()
        
        # Reset acceptance counters
        self.acceptance_count = 0
        self.total_proposals = 0
        
        for i in range(1, n_samples + self.burn_in):
            # Propose new state
            proposal = np.random.multivariate_normal(
                samples[i-1], proposal_cov)
            
            # Compute acceptance ratio
            log_ratio = self._log_target(proposal) - self._log_target(samples[i-1])
            
            # Accept or reject
            self.total_proposals += 1
            if np.log(np.random.random()) < log_ratio:
                samples[i] = proposal
                self.acceptance_count += 1
            else:
                samples[i] = samples[i-1]

            # Adapt step size during burn-in
            if i % 500 == 0 and i < self.burn_in:
                acceptance_rate = self.acceptance_count / self.total_proposals
                if acceptance_rate < 0.2:
                    self.step_size *= 0.75  # Reduce step size
                elif acceptance_rate > 0.4:
                    self.step_size *= 1.25  # Increase step size
        
        # Print acceptance rate
        acceptance_rate = self.acceptance_count / self.total_proposals
        print(f"MH acceptance rate: {acceptance_rate:.2f} and step size: {self.step_size:.2f}")
        
        # Update current state for next call
        self.current_state = samples[-1]
        
        # Discard burn-in and thin the chain
        final_samples = samples[self.burn_in:]
        
        # Check if chain is well-mixed
        if acceptance_rate < 0.1:
            print("Warning: Low acceptance rate - consider decreasing step_size")
        elif acceptance_rate > 0.7:
            print("Warning: High acceptance rate - consider increasing step_size")
            
        return final_samples