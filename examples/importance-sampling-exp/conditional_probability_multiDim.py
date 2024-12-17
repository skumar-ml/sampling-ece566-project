import numpy as np
from distributions.implementations import GaussianMV
from samplers.implementations import UnitCubeSampler, SobolSampler, TruncatedMHSampler
from main import run_sampling_experiment
import matplotlib.pyplot as plt
from scipy.stats import norm
from typing import Tuple, List

def generate_random_cov(dim: int, correlation_strength: float = 0.8) -> np.ndarray:
    """
    Generate a random positive definite covariance matrix.
    
    Args:
        dim: Dimension of the covariance matrix
        correlation_strength: Controls the off-diagonal correlation strength
    """
    # Generate random correlation matrix
    A = np.random.randn(dim, dim)
    cov = np.dot(A, A.transpose())
    
    # Normalize to correlation matrix
    D = np.sqrt(np.diag(np.diag(cov)))
    corr = np.linalg.inv(D) @ cov @ np.linalg.inv(D)
    
    # Scale off-diagonal elements
    corr = corr * correlation_strength
    np.fill_diagonal(corr, 1.0)
    
    # Even tighter variances
    variances = np.random.uniform(0.1, 0.3, size=dim)  # Reduced upper bound
    D = np.sqrt(np.diag(variances))
    return D @ corr @ D

def get_conditional_distribution(x1_fixed: float, mean: np.ndarray, 
                               cov: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute conditional mean and covariance for X_{2:n}|X_1=x1_fixed
    
    Args:
        x1_fixed: Fixed value for first dimension
        mean: Original mean vector
        cov: Original covariance matrix
    Returns:
        cond_mean: Mean vector of conditional distribution
        cond_cov: Covariance matrix of conditional distribution
    """
    # Split mean and covariance
    mu1 = mean[0]
    mu2 = mean[1:]
    sigma11 = cov[0, 0].reshape(1, 1)  # Make 2D: (1,1)
    sigma12 = cov[0, 1:].reshape(1, -1)  # Make 2D: (1,n-1)
    sigma21 = cov[1:, 0].reshape(-1, 1)  # Make 2D: (n-1,1)
    sigma22 = cov[1:, 1:]  # Already 2D: (n-1,n-1)
    
    # Compute conditional parameters
    cond_mean = mu2 + (sigma21 @ np.linalg.inv(sigma11) @ (x1_fixed - mu1).reshape(1, 1)).flatten()
    cond_cov = sigma22 - sigma21 @ np.linalg.inv(sigma11) @ sigma12
    
    return cond_mean, cond_cov

def get_thresholds(cond_mean: np.ndarray, cond_cov: np.ndarray, 
                  target_prob: float = 0.8) -> np.ndarray:  # Increased target probability
    """
    Compute thresholds for each dimension to achieve target marginal probability.
    """
    std_diag = np.sqrt(np.diag(cond_cov))
    z_score = norm.ppf(1 - target_prob)
    # Move thresholds even closer to mean
    return cond_mean + 0.25 * z_score * std_diag  # Reduced factor from 0.5 to 0.25

def compute_ground_truth(cond_mean: np.ndarray, cond_cov: np.ndarray, 
                        thresholds: np.ndarray, n_mc: int = 100000) -> float:
    """
    Compute ground truth probability using Monte Carlo integration.
    P(X_2 > t_2, ..., X_n > t_n | X_1 = x1_fixed)
    """
    samples = np.random.multivariate_normal(cond_mean, cond_cov, size=n_mc)
    indicator = np.all(samples > thresholds, axis=1)
    return np.mean(indicator)

def get_proposal_distribution(cond_mean: np.ndarray, cond_cov: np.ndarray, 
                            thresholds: np.ndarray) -> GaussianMV:
    """
    Creates a proposal distribution for importance sampling.
    Strategy:
    1. Shift mean towards the region of interest (beyond thresholds)
    2. Inflate covariance to get heavier tails
    3. Maintain correlation structure
    """
    # Compute the direction from mean to threshold
    direction = thresholds - cond_mean
    
    # Shift mean slightly beyond thresholds
    proposal_mean = thresholds + 1 * direction
    
    # Inflate covariance while maintaining correlation structure
    inflation_factor = 1.5  # Adjust this for better coverage
    proposal_cov = inflation_factor * cond_cov
    
    return GaussianMV(mean=proposal_mean, cov=proposal_cov)

def run_dimension_experiment(dims: List[int], n_samples: int):
    """Run experiment for each dimension and collect results."""
    results = []
    x1_fixed = 0.0
    target_prob = 0.8
    
    for dim in dims:
        print(f"\nRunning experiment for dimension {dim}")
        
        # Generate random parameters for full distribution
        mean = np.zeros(dim)
        cov = generate_random_cov(dim)
        
        # Get conditional distribution
        cond_mean, cond_cov = get_conditional_distribution(x1_fixed, mean, cov)
        
        # Compute thresholds and ground truth
        thresholds = get_thresholds(cond_mean, cond_cov, target_prob)
        true_prob = compute_ground_truth(cond_mean, cond_cov, thresholds)
        
        # Create target distribution
        target_dist = GaussianMV(mean=cond_mean, cov=cond_cov)
        
        # Setup estimators
        mc_estimator = multivariate_probability_estimator(thresholds)
        qmc_estimator = multivariate_probability_estimator(thresholds)
        mh_estimator = optimal_proposal_estimator(target_dist)
        
        # Setup samplers
        mc_sampler = UnitCubeSampler()
        qmc_sampler = SobolSampler(scramble=True)
        mh_sampler = TruncatedMHSampler(thresholds=thresholds, step_size=0.25, burn_in=1000)
        
        methods = [
            (target_dist, mc_estimator, mc_sampler, "MC"),
            (target_dist, qmc_estimator, qmc_sampler, "QMC"),
            (target_dist, mh_estimator, mh_sampler, "MH")
        ]
        
        dim_results = {}
        for dist, estimator, sampler, name in methods:
            exp_results = run_sampling_experiment(
                distribution=dist,
                target_function=estimator,
                sampler=sampler,
                n_samples=n_samples,
                n_dimensions=dim-1
            )
            final_estimate = exp_results['convergence_data'].running_means[-1]
            relative_error = abs(final_estimate - true_prob) / true_prob
            dim_results[name] = relative_error
            dim_results[f"{name}_prob"] = final_estimate  # Store the probability estimate
        
        dim_results['dimension'] = dim
        dim_results['true_prob'] = true_prob
        results.append(dim_results)
    
    return results

def multivariate_probability_estimator(thresholds: np.ndarray):
    """Returns a function that estimates P(X > thresholds)"""
    def estimator(x: np.ndarray) -> float:
        """
        Args:
            x: Array of shape (n_samples,) or (n_samples, n_dim)
        Returns:
            Array of shape (n_samples,) with 1 where all dimensions exceed thresholds
        """
        # Expand dimensions if x is 1D
        if x.ndim == 1:
            x = x.reshape(-1, 1)
        
        # Broadcast thresholds if needed
        if thresholds.ndim == 1:
            thresholds_expanded = thresholds.reshape(1, -1)
        else:
            thresholds_expanded = thresholds
            
        return np.all(x > thresholds_expanded, axis=1).astype(float)
    return estimator

def importance_sampling_estimator(thresholds: np.ndarray, 
                                target_dist: GaussianMV, 
                                proposal_dist: GaussianMV):
    """Returns an importance sampling estimator function"""
    def estimator(x: np.ndarray) -> float:
        """
        Args:
            x: Array of shape (n_samples,) or (n_samples, n_dim)
        Returns:
            Array of shape (n_samples,) with importance weights
        """
        # Expand dimensions if x is 1D
        if x.ndim == 1:
            x = x.reshape(-1, 1)
            
        # Broadcast thresholds if needed
        if thresholds.ndim == 1:
            thresholds_expanded = thresholds.reshape(1, -1)
        else:
            thresholds_expanded = thresholds
            
        indicator = np.all(x > thresholds_expanded, axis=1).astype(float)
        weights = target_dist.pdf(x) / proposal_dist.pdf(x)
        return indicator * weights
    return estimator

def optimal_proposal_estimator(target_dist: GaussianMV):
    """
    Returns an estimator for samples from the optimal proposal distribution.
    For p*(x) ∝ 1(x > thresholds) * target_dist.pdf(x)
    
    The normalizing constant Z = P(X > thresholds) is what we're trying to estimate.
    """
    def estimator(x: np.ndarray) -> float:
        """
        Args:
            x: Array of shape (n_samples, n_dim) - samples from MH
        Returns:
            Array of shape (n_samples,) with weights
        """
        return target_dist.pdf(x)
    
    return estimator

def plot_dimension_results(results: List[dict]):
    """Plot relative errors vs dimension for each method."""
    dims = [r['dimension'] for r in results]
    methods = ['MC', 'QMC', 'MH']
    method_names = {
        'MC': 'Monte Carlo',
        'QMC': 'Quasi-Monte Carlo',
        'MH': 'Metropolis-Hastings'
    }
    colors = {
        'MC': 'blue',
        'QMC': 'orange',
        'MH': 'green'
    }
    
    plt.figure(figsize=(12, 8))
    
    for method in methods:
        errors = [r[method] for r in results]
        plt.plot(dims, errors, marker='o', label=method_names[method], 
                color=colors[method], linewidth=2)
    
    plt.xlabel('Dimension')
    plt.ylabel('Relative Error')
    plt.title('Relative Error vs Dimension by Method')
    plt.grid(True)
    plt.legend()
    plt.yscale('log')
    plt.tight_layout()
    plt.savefig('examples/importance-sampling-exp/dimension_comparison.png')
    plt.show()

if __name__ == "__main__":
    # Parameters
    dimensions = list(range(3, 15))
    n_samples = 2**14
    
    # Run experiments
    results = run_dimension_experiment(dimensions, n_samples)
    
    # Print results
    print("\nResults by dimension:")
    print("-" * 70)
    for r in results:
        dim = r['dimension']
        print(f"\nDimension {dim}:")
        print(f"True probability: {r['true_prob']:.6e}")
        print("Method Results:")
        print("  Method  Probability    Rel. Error")
        print("  " + "-" * 30)
        for method in ['MC', 'QMC', 'MH']:
            # Calculate probability from relative error and true probability
            prob = r['true_prob'] * (1 + r[method]) if r[method] > 0 else r['true_prob'] * (1 - r[method])
            print(f"  {method:<7} {prob:.6e}  {r[method]:.6e}")
    
    # Plot results
    plot_dimension_results(results)