from abc import ABC, abstractmethod
import numpy as np

class Distribution(ABC):
    """Base class for probability distributions."""
    
    @abstractmethod
    def pdf(self, x: np.ndarray) -> float:
        """Compute probability density at point x."""
        pass
    
    @abstractmethod
    def log_pdf(self, x: np.ndarray) -> float:
        """Compute log probability density at point x."""
        pass 