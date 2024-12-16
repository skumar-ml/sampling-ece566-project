import numpy as np
from scipy.stats import norm, multivariate_normal
from .base import Distribution

class Gaussian1D(Distribution):
    def __init__(self, mean: float, std: float):
        """
        1D Gaussian distribution.
        
        Args:
            mean: Mean of the distribution
            std: Standard deviation
        """
        self.mean = mean
        self.std = std
        self.dist = norm(mean, std)
    
    def sample(self, n_samples: int) -> np.ndarray:
        return self.dist.rvs(size=n_samples)
    
    def pdf(self, x: np.ndarray) -> np.ndarray:
        return self.dist.pdf(x)
    
    def log_pdf(self, x: np.ndarray) -> np.ndarray:
        return self.dist.logpdf(x)
    
    def inverse_cdf(self, u: np.ndarray) -> np.ndarray:
        """Transform uniform samples to this distribution."""
        return self.dist.ppf(u)

class GaussianMV(Distribution):
    def __init__(self, mean: np.ndarray, cov: np.ndarray):
        """
        Multivariate Gaussian distribution.
        
        Args:
            mean: Mean vector
            cov: Covariance matrix
        """
        self.mean = mean
        self.cov = cov
        self.dist = multivariate_normal(mean=mean, cov=cov, allow_singular=True)
        
        # Compute Cholesky decomposition for inverse transform
        self.L = np.linalg.cholesky(cov)
    
    def sample(self, n_samples: int) -> np.ndarray:
        return self.dist.rvs(size=n_samples)
    
    def pdf(self, x: np.ndarray) -> np.ndarray:
        return self.dist.pdf(x)
    
    def log_pdf(self, x: np.ndarray) -> np.ndarray:
        return self.dist.logpdf(x)
    
    def inverse_cdf(self, u: np.ndarray) -> np.ndarray:
        """
        Transform uniform samples to this distribution using Box-Muller
        and linear transformation.
        
        Args:
            u: Uniform samples of shape (n_samples, n_dimensions)
        """
        # Convert uniform to standard normal using Box-Muller
        n_samples = u.shape[0]
        z = np.zeros_like(u)
        
        # Handle pairs of dimensions
        for i in range(0, u.shape[1] - 1, 2):
            r = np.sqrt(-2 * np.log(u[:, i]))
            theta = 2 * np.pi * u[:, i+1]
            z[:, i] = r * np.cos(theta)
            z[:, i+1] = r * np.sin(theta)
        
        # Handle last dimension if odd
        if u.shape[1] % 2:
            last = u.shape[1] - 1
            r = np.sqrt(-2 * np.log(u[:, last]))
            theta = 2 * np.pi * u[:, 0]  # Reuse first column
            z[:, last] = r * np.cos(theta)
        
        # Transform to desired mean and covariance
        return self.mean + z @ self.L.T

class UniformCube(Distribution):
    """Uniform distribution over a unit cube [0,1]^n."""
    
    def __init__(self, n_dimensions: int):
        self.n_dimensions = n_dimensions
    
    def pdf(self, x: np.ndarray) -> float:
        # Check if point is inside unit cube
        inside = np.all((x >= 0) & (x <= 1))
        return 1.0 if inside else 0.0
    
    def log_pdf(self, x: np.ndarray) -> float:
        inside = np.all((x >= 0) & (x <= 1))
        return 0.0 if inside else -np.inf
    
    def inverse_cdf(self, u: np.ndarray) -> np.ndarray:
        """
        Transform uniform samples to this distribution.
        For uniform cube, this is just the identity function.
        
        Args:
            u: Uniform samples of shape (n_samples, n_dimensions)
            
        Returns:
            The same samples unchanged since they're already uniform in [0,1]
        """
        if u.shape[1] != self.n_dimensions:
            raise ValueError(f"Input shape {u.shape} does not match distribution dimensions {self.n_dimensions}")
        return u