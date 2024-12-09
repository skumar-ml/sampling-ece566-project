import numpy as np
from .base import Distribution

class MultivariateNormal(Distribution):
    """Multivariate normal distribution."""
    
    def __init__(self, mean: np.ndarray = None, cov: np.ndarray = None):
        self.mean = mean
        self.cov = cov
        self.inv_cov = None
        self.norm_const = None
        
    def setup(self, n_dimensions: int):
        if self.mean is None:
            self.mean = np.zeros(n_dimensions)
        if self.cov is None:
            self.cov = np.eye(n_dimensions)
        self.inv_cov = np.linalg.inv(self.cov)
        self.norm_const = (
            (2 * np.pi) ** (-n_dimensions/2) * 
            np.linalg.det(self.cov) ** (-0.5)
        )
    
    def pdf(self, x: np.ndarray) -> float:
        diff = x - self.mean
        return self.norm_const * np.exp(
            -0.5 * diff.T @ self.inv_cov @ diff
        )
    
    def log_pdf(self, x: np.ndarray) -> float:
        diff = x - self.mean
        return (
            np.log(self.norm_const) - 
            0.5 * diff.T @ self.inv_cov @ diff
        )

class Mixture(Distribution):
    """Mixture of distributions."""
    
    def __init__(self, distributions: list, weights: np.ndarray):
        self.distributions = distributions
        self.weights = weights / np.sum(weights)
    
    def pdf(self, x: np.ndarray) -> float:
        return np.sum([
            w * d.pdf(x) 
            for w, d in zip(self.weights, self.distributions)
        ])
    
    def log_pdf(self, x: np.ndarray) -> float:
        return np.log(self.pdf(x)) 

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