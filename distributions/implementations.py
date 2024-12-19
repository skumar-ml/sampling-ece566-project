import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from .base import Distribution
import pandas as pd
from scipy.stats import norm, multivariate_normal

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
        # Convert uniform to standard normal using inverse normal CDF
        z = norm.ppf(u)
        
        # Transform to desired mean and covariance using Cholesky decomposition
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

class GMMDistribution(Distribution):
    """Gaussian Mixture Model distribution for medical cost data."""
    
    def __init__(self, n_components: int = 3):
        """
        Args:
            n_components: Number of Gaussian components in the mixture
        """
        self.n_components = n_components
        self.gmm = None
        self.feature_names = None
        self.n_dimensions = None
    
    def sample(self, n_samples: int) -> np.ndarray:
        """Generate samples from the distribution."""
        if self.gmm is None:
            raise RuntimeError("Must call fit() before sampling")
        
        # Sample directly in scaled space
        X_scaled, _ = self.gmm.sample(n_samples)
        return X_scaled
    
    def pdf(self, x: np.ndarray) -> np.ndarray:
        """Compute probability density."""
        if self.gmm is None:
            raise RuntimeError("Must call fit() before computing pdf")
        return np.exp(self.log_pdf(x))
    
    def log_pdf(self, x: np.ndarray) -> np.ndarray:
        """Compute log probability density."""
        if self.gmm is None:
            raise RuntimeError("Must call fit() before computing log_pdf")

        # If x.ndim == 1, add a dimension
        if x.ndim == 1:
            x = x[np.newaxis, :]

        return self.gmm.score_samples(x)
    
    def fit(self, X_scaled: np.ndarray):
        """
        Fit the GMM to scaled data.
        
        Args:
            X_scaled: Array of scaled features
            feature_names: Optional parameter, not used
        """
        self.n_dimensions = X_scaled.shape[1]
        
        # Fit GMM directly to scaled data
        self.gmm = GaussianMixture(
            n_components=self.n_components,
            covariance_type='full',
            random_state=42
        )
        self.gmm.fit(X_scaled)
    
    @property
    def weights(self) -> np.ndarray:
        """Get mixture weights."""
        if self.gmm is None:
            raise RuntimeError("Must call fit() before accessing weights")
        return self.gmm.weights_
    
    @property
    def means(self) -> np.ndarray:
        """Get component means."""
        if self.gmm is None:
            raise RuntimeError("Must call fit() before accessing means")
        return self.gmm.means_
    
    @property
    def covariances(self) -> np.ndarray:
        """Get component covariances."""
        if self.gmm is None:
            raise RuntimeError("Must call fit() before accessing covariances")
        return self.gmm.covariances_
    
    def inverse_cdf(self, u: np.ndarray) -> np.ndarray:
        """
        Transform uniform samples to GMM samples in scaled space.
        
        Args:
            u: Uniform samples of shape (n_samples, n_dimensions + 1)
                Last dimension is used for component selection
            
        Returns:
            Array of samples in scaled space
        """
        if self.gmm is None:
            raise RuntimeError("Must call fit() before inverse transform")
            
        n_samples = u.shape[0]
        
        # Separate uniform samples
        uniform_samples = u[:, :-1]  # For Gaussian sampling
        component_selector = u[:, -1]  # For selecting mixture component
        
        # Initialize output array
        samples_scaled = np.zeros((n_samples, self.n_dimensions))
        
        # Cumulative weights for component selection
        cumulative_weights = np.cumsum(self.weights)
        
        # Generate samples
        for i in range(n_samples):
            # Select component
            component_idx = np.searchsorted(cumulative_weights, component_selector[i])
            component_idx = min(component_idx, len(self.weights) - 1)  # Handle edge case
            
            # Get component parameters
            mu = self.means[component_idx]
            cov = self.covariances[component_idx]
            
            # Transform uniform to standard normal
            std_normal = norm.ppf(uniform_samples[i])
            
            # Transform to component distribution
            L = np.linalg.cholesky(cov)
            samples_scaled[i] = mu + np.dot(L, std_normal)
        
        return samples_scaled
    
    @property
    def cov(self) -> np.ndarray:
        """
        Returns weighted average of component covariances.
        This is a rough approximation of the overall covariance structure.
        """
        if self.gmm is None:
            raise RuntimeError("Must call fit() before accessing covariance")
            
        # Compute weighted average of covariances
        avg_cov = np.zeros_like(self.gmm.covariances_[0])
        for w, cov in zip(self.gmm.weights_, self.gmm.covariances_):
            avg_cov += w * cov
            
        return avg_cov