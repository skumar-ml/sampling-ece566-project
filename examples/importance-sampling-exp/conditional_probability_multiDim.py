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

def run_dimension_experiment(dims: List[int], n_samples: int):
    """Run experiment for each dimension and collect results."""
    results = []
    x1_fixed = 0.0
    target_prob = 0.8
    
    for dim in dims:
        print(f"\nRunning experiment for dimension {dim}")
        
        mean = np.zeros(dim)
        cov = generate_random_cov(dim)
        cond_mean, cond_cov = get_conditional_distribution(x1_fixed, mean, cov)
        thresholds = get_thresholds(cond_mean, cond_cov, target_prob)
        true_prob = compute_ground_truth(cond_mean, cond_cov, thresholds, n_mc=10**7)
        
        target_dist = GaussianMV(mean=cond_mean, cov=cond_cov)
        
        mc_estimator = multivariate_probability_estimator(thresholds)
        qmc_estimator = multivariate_probability_estimator(thresholds)
        
        kde_samples, kde = mh_kde_importance_sampling(
            target_dist=target_dist,
            thresholds=thresholds,
            n_samples=n_samples,
            n_mh_samples=n_samples
        )
        mh_kde_is_est = mh_kde_is_estimator(target_dist, kde, thresholds)
        
        mc_sampler = MonteCarloSampler()
        qmc_sampler = SobolSampler(scramble=True)
        
        methods = [
            (target_dist, mc_estimator, mc_sampler, "MC"),
            (target_dist, qmc_estimator, qmc_sampler, "QMC"),
            (None, mh_kde_is_est, lambda n: kde_samples, "MH-KDE-IS")
        ]
        
        dim_results = {}
        for dist, estimator, sampler, name in methods:
            if name == "MH-KDE-IS":
                weights = estimator(kde_samples)
                estimate = np.mean(weights)
                relative_error = abs(estimate - true_prob) / true_prob
            else:
                exp_results = run_sampling_experiment(
                    distribution=dist,
                    target_function=estimator,
                    sampler=sampler,
                    n_samples=n_samples,
                    n_dimensions=dim-1
                )
                estimate = exp_results['convergence_data'].running_means[-1]
                relative_error = abs(estimate - true_prob) / true_prob
            
            dim_results[name] = relative_error * 100
            dim_results[f"{name}_prob"] = estimate
        
        dim_results['dimension'] = dim
        dim_results['true_prob'] = true_prob
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

if __name__ == "__main__":
    dimensions = list(range(3, 15))
    n_samples = 2**14
    
    results = run_dimension_experiment(dimensions, n_samples)
    
    print("\nResults by dimension:")
    print("-" * 70)
    for r in results:
        dim = r['dimension']
        print(f"\nDimension {dim}:")
        print(f"True probability: {r['true_prob']:.6e}")
        print("Method Results:")
        print("  Method  Probability    Rel. Error")
        print("  " + "-" * 30)
        for method in ['MC', 'QMC', 'MH-KDE-IS']:
            print(f"  {method:<7} {r[f'{method}_prob']:.6e}  {r[method]:.6e}")
    
    plot_dimension_results(results)