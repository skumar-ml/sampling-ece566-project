import numpy as np
from distributions.implementations import GaussianMV
from samplers.implementations import MonteCarloSampler, SobolSampler, TruncatedMHSampler
from main import run_sampling_experiment
import matplotlib.pyplot as plt
from scipy.stats import norm, gaussian_kde
from typing import Tuple, List

def generate_random_cov(dim: int, correlation_strength: float = 0.8) -> np.ndarray:
    """Generate a random positive definite covariance matrix."""
    A = np.random.randn(dim, dim)
    cov = np.dot(A, A.transpose())
    
    D = np.sqrt(np.diag(np.diag(cov)))
    corr = np.linalg.inv(D) @ cov @ np.linalg.inv(D)
    
    corr = corr * correlation_strength
    np.fill_diagonal(corr, 1.0)
    
    variances = np.random.uniform(0.1, 0.3, size=dim)
    D = np.sqrt(np.diag(variances))
    return D @ corr @ D

def get_conditional_distribution(x1_fixed: float, mean: np.ndarray, 
                               cov: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Compute conditional mean and covariance for X_{2:n}|X_1=x1_fixed"""
    mu1 = mean[0]
    mu2 = mean[1:]
    sigma11 = cov[0, 0].reshape(1, 1)
    sigma12 = cov[0, 1:].reshape(1, -1)
    sigma21 = cov[1:, 0].reshape(-1, 1)
    sigma22 = cov[1:, 1:]
    
    cond_mean = mu2 + (sigma21 @ np.linalg.inv(sigma11) @ (x1_fixed - mu1).reshape(1, 1)).flatten()
    cond_cov = sigma22 - sigma21 @ np.linalg.inv(sigma11) @ sigma12
    
    return cond_mean, cond_cov

def get_thresholds(cond_mean: np.ndarray, cond_cov: np.ndarray, 
                  target_prob: float = 0.8) -> np.ndarray:
    """Compute thresholds for each dimension."""
    std_diag = np.sqrt(np.diag(cond_cov))
    z_score = norm.ppf(1 - target_prob)
    return cond_mean + 0.25 * z_score * std_diag

def compute_ground_truth(cond_mean: np.ndarray, cond_cov: np.ndarray, 
                        thresholds: np.ndarray, n_mc: int = 100000) -> float:
    """Compute ground truth probability using Monte Carlo integration."""
    samples = np.random.multivariate_normal(cond_mean, cond_cov, size=n_mc)
    indicator = np.all(samples > thresholds, axis=1)
    return np.mean(indicator)

def mh_kde_importance_sampling(target_dist: GaussianMV, thresholds: np.ndarray, 
                             n_samples: int, n_mh_samples: int = 10000):
    """Performs MH-KDE-IS sampling targeting the optimal distribution."""
    def optimal_log_density(x):
        # Log density of target * indicator
        if np.all(x > thresholds):
            return target_dist.log_pdf(x)
        return -np.inf
    
    mh_sampler = TruncatedMHSampler(
        thresholds=thresholds, 
        step_size=0.1, 
        burn_in=2000,
        log_target_density=optimal_log_density
    )
    mh_sampler.setup(target_dist, len(thresholds))
    mh_samples = mh_sampler.generate_samples(n_mh_samples)
    
    # Only use valid samples for KDE
    valid_samples = mh_samples[np.all(mh_samples > thresholds, axis=1)]
    if len(valid_samples) < 10:
        raise ValueError("Too few valid samples for KDE estimation")
    
    kde = gaussian_kde(valid_samples.T)
    kde_samples = kde.resample(n_samples).T
    
    return kde_samples, kde

def mh_kde_is_estimator(target_dist: GaussianMV, kde: gaussian_kde, thresholds: np.ndarray):
    """Returns an estimator for MH-KDE-IS method."""
    def estimator(x: np.ndarray) -> np.ndarray:
        indicator = np.all(x > thresholds, axis=1).astype(float)
        target_density = target_dist.pdf(x)
        proposal_density = kde.evaluate(x.T)
        weights = indicator * target_density / proposal_density
        return weights
    return estimator

def multivariate_probability_estimator(thresholds: np.ndarray):
    """Returns a function that estimates P(X > thresholds)"""
    def estimator(x: np.ndarray) -> float:
        if x.ndim == 1:
            x = x.reshape(-1, 1)
        if thresholds.ndim == 1:
            thresholds_expanded = thresholds.reshape(1, -1)
        else:
            thresholds_expanded = thresholds
        return np.all(x > thresholds_expanded, axis=1).astype(float)
    return estimator

def run_single_dimension_experiment(dim: int, n_samples: int):
    """Run experiment for a single dimension and return results."""
    # Setup the problem
    mean = np.zeros(dim)
    cov = generate_random_cov(dim)
    cond_mean, cond_cov = get_conditional_distribution(0.0, mean, cov)
    thresholds = get_thresholds(cond_mean, cond_cov)
    true_prob = compute_ground_truth(cond_mean, cond_cov, thresholds)
    
    target_dist = GaussianMV(mean=cond_mean, cov=cond_cov)
    
    # Setup estimators and samplers
    mc_estimator = multivariate_probability_estimator(thresholds)
    qmc_estimator = multivariate_probability_estimator(thresholds)
    
    mc_sampler = MonteCarloSampler()
    qmc_sampler = SobolSampler(scramble=True)
    
    # Run standard MC and QMC experiments
    mc_results = run_sampling_experiment(
        distribution=target_dist,
        target_function=mc_estimator,
        sampler=mc_sampler,
        n_samples=n_samples,
        n_dimensions=dim-1
    )
    
    qmc_results = run_sampling_experiment(
        distribution=target_dist,
        target_function=qmc_estimator,
        sampler=qmc_sampler,
        n_samples=n_samples,
        n_dimensions=dim-1
    )
    
    # Run MH-KDE-IS experiment
    kde_samples, kde = mh_kde_importance_sampling(
        target_dist=target_dist,
        thresholds=thresholds,
        n_samples=n_samples,
        n_mh_samples=n_samples
    )
    mh_kde_is_est = mh_kde_is_estimator(target_dist, kde, thresholds)
    
    # Calculate running means for MH-KDE-IS
    weights = mh_kde_is_est(kde_samples)
    running_means = np.cumsum(weights) / np.arange(1, len(weights) + 1)
    
    return {
        'mc_results': mc_results,
        'qmc_results': qmc_results,
        'mh_kde_is_means': running_means,
        'true_prob': true_prob
    }

def run_dimension_experiment(dims: List[int], n_samples: int):
    """Run experiment for each dimension and collect results."""
    results = []
    
    for dim in dims:
        print(f"\nRunning experiment for dimension {dim}")
        
        # Run experiment for this dimension
        dim_exp_results = run_single_dimension_experiment(dim, n_samples)
        
        # Calculate relative errors for each method
        true_prob = dim_exp_results['true_prob']
        dim_results = {
            'dimension': dim,
            'true_prob': true_prob
        }
        
        # MC relative error
        mc_estimate = dim_exp_results['mc_results']['convergence_data'].running_means[-1]
        dim_results['MC'] = abs(mc_estimate - true_prob) / true_prob * 100
        dim_results['MC_prob'] = mc_estimate
        
        # QMC relative error
        qmc_estimate = dim_exp_results['qmc_results']['convergence_data'].running_means[-1]
        dim_results['QMC'] = abs(qmc_estimate - true_prob) / true_prob * 100
        dim_results['QMC_prob'] = qmc_estimate
        
        # MH-KDE-IS relative error
        mh_kde_estimate = dim_exp_results['mh_kde_is_means'][-1]
        dim_results['MH-KDE-IS'] = abs(mh_kde_estimate - true_prob) / true_prob * 100
        dim_results['MH-KDE-IS_prob'] = mh_kde_estimate
        
        results.append(dim_results)
    
    return results

def plot_dimension_results(results: List[dict]):
    """Plot relative errors vs dimension for each method."""
    dims = [r['dimension'] for r in results]
    methods = ['MC', 'QMC', 'MH-KDE-IS']
    method_names = {
        'MC': 'Monte Carlo',
        'QMC': 'Quasi-Monte Carlo',
        'MH-KDE-IS': 'MH-KDE-IS'
    }
    colors = {
        'MC': 'blue',
        'QMC': 'orange',
        'MH-KDE-IS': 'green'
    }
    
    plt.figure(figsize=(12, 8))
    
    for method in methods:
        errors = [r[method] for r in results]
        plt.plot(dims, errors, marker='o', label=method_names[method], 
                color=colors[method], linewidth=2)
    
    plt.xlabel('Dimension')
    plt.ylabel('Relative Percentage Error')
    plt.title('Relative Error vs Dimension by Method')
    plt.grid(True)
    plt.legend()
    plt.yscale('log')
    plt.tight_layout()
    plt.savefig('examples/importance-sampling-exp/dimension_comparison.png')
    plt.show()

def plot_convergence_for_dimensions(dims: List[int], n_samples: int):
    """Plot convergence for selected dimensions in a subplot."""
    # Select first, middle and last dimension
    selected_dims = [dims[0], dims[len(dims)//2], dims[-1]]
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('Convergence Analysis by Dimension', fontsize=14)
    
    colors = {
        'MC': 'blue',
        'QMC': 'orange',
        'MH-KDE-IS': 'green'
    }
    
    for idx, dim in enumerate(selected_dims):
        print(f"\nGenerating convergence plot for dimension {dim}")
        
        results = run_single_dimension_experiment(dim, n_samples)
        
        # Plot results
        ax = axes[idx]
        sample_points = np.arange(1, n_samples + 1)
        
        ax.plot(sample_points, results['mc_results']['convergence_data'].running_means, 
                label='MC', color=colors['MC'], alpha=0.8)
        ax.plot(sample_points, results['qmc_results']['convergence_data'].running_means, 
                label='QMC', color=colors['QMC'], alpha=0.8)
        ax.plot(sample_points, results['mh_kde_is_means'], 
                label='MH-KDE-IS', color=colors['MH-KDE-IS'], alpha=0.8)
        ax.axhline(y=results['true_prob'], color='r', linestyle='--', label='True Value')
        
        ax.set_title(f'Dimension {dim}')
        ax.set_xlabel('Number of Samples')
        ax.set_ylabel('Probability Estimate')
        ax.grid(True, alpha=0.3)
        ax.legend()
        ax.set_xscale('log')
    
    plt.tight_layout()
    plt.savefig('examples/importance-sampling-exp/convergence_comparison_multiDim.png')
    plt.show()

if __name__ == "__main__":
    dimensions = list(range(3, 15))
    n_samples = 2**14
    
    # Run main dimension experiment
    results = run_dimension_experiment(dimensions, n_samples)
    plot_dimension_results(results)
    
    # Generate convergence plots
    plot_convergence_for_dimensions(dimensions, n_samples)