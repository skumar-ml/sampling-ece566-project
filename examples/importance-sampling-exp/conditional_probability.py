import numpy as np
from distributions.implementations import Gaussian1D, GaussianMV
from samplers.implementations import UnitCubeSampler, SobolSampler
from main import run_sampling_experiment
import matplotlib.pyplot as plt
from scipy.stats import norm

def get_conditional_distribution(x1_fixed: float) -> Gaussian1D:
    """
    Creates the conditional distribution X2|X1=x1_fixed.
    For a 2D Gaussian, the conditional is 1D Gaussian with:
    mean = μ2 + ρ(σ2/σ1)(x1 - μ1)
    std = sqrt(σ2^2(1 - ρ^2))
    """
    # Original distribution parameters
    mean = np.array([1.0, 0.0])
    cov = np.array([[2.0, 0.5],
                    [0.5, 1.0]])
    
    # Compute conditional parameters
    cond_mean = mean[1] + (cov[1,0]/cov[0,0]) * (x1_fixed - mean[0])
    cond_std = np.sqrt(cov[1,1] - (cov[1,0]**2)/cov[0,0])
    
    return Gaussian1D(mean=cond_mean, std=cond_std)

def get_proposal_distribution(target_dist: Gaussian1D, threshold: float) -> Gaussian1D:
    """Creates a proposal distribution for importance sampling."""
    # Shift the mean to threshold but keep same standard deviation
    return Gaussian1D(mean=threshold, std=target_dist.std)

def conditional_probability_estimator(threshold: float):
    """Returns a function that estimates P(X2 > threshold | X1 = x1_fixed)"""
    
    def estimator(x: np.ndarray) -> float:
        """
        Args:
            x: Array of shape (n_samples,) for 1D samples
        Returns:
            Array of shape (n_samples,) with 1 where x > threshold, 0 otherwise
        """
        return (x > threshold).astype(float)
    
    return estimator

def importance_sampling_estimator(threshold: float, 
                                target_dist: Gaussian1D, 
                                proposal_dist: Gaussian1D):
    """Returns an importance sampling estimator function"""
    
    def estimator(x: np.ndarray) -> float:
        """
        Args:
            x: Array of shape (n_samples,) for 1D samples
        Returns:
            Array of shape (n_samples,) with importance weights
        """
        indicator = (x > threshold).astype(float)
        weights = target_dist.pdf(x) / proposal_dist.pdf(x)
        return indicator * weights
    
    return estimator

def run_comparison(n_samples: int, x1_fixed: float, threshold: float):
    # Setup distributions
    target_dist = get_conditional_distribution(x1_fixed)
    proposal_dist = get_proposal_distribution(target_dist, threshold)
    
    # Setup estimators
    mc_estimator = conditional_probability_estimator(threshold)
    qmc_estimator = conditional_probability_estimator(threshold)
    is_estimator = importance_sampling_estimator(threshold, target_dist, proposal_dist)
    is_qmc_estimator = importance_sampling_estimator(threshold, target_dist, proposal_dist)
    
    # Run experiments
    mc_sampler = UnitCubeSampler()
    qmc_sampler = SobolSampler(scramble=True)
    
    mc_results = run_sampling_experiment(
        distribution=target_dist,
        target_function=mc_estimator,
        sampler=mc_sampler,
        n_samples=n_samples,
        n_dimensions=1  # Now 1D
    )
    
    qmc_results = run_sampling_experiment(
        distribution=target_dist,
        target_function=qmc_estimator,
        sampler=qmc_sampler,
        n_samples=n_samples,
        n_dimensions=1  # Now 1D
    )
    
    is_results = run_sampling_experiment(
        distribution=proposal_dist,
        target_function=is_estimator,
        sampler=mc_sampler,
        n_samples=n_samples,
        n_dimensions=1  # Now 1D
    )
    
    is_qmc_results = run_sampling_experiment(
        distribution=proposal_dist,
        target_function=is_qmc_estimator,
        sampler=qmc_sampler,
        n_samples=n_samples,
        n_dimensions=1
    )
    
    return mc_results, qmc_results, is_results, is_qmc_results

def visualize_distributions(target_dist: Gaussian1D, proposal_dist: Gaussian1D, 
                          threshold: float):
    """
    Creates a visualization of the conditional and proposal distributions.
    """
    # Create points for plotting
    x = np.linspace(target_dist.mean - 4*target_dist.std, 
                    proposal_dist.mean + 4*proposal_dist.std, 1000)
    
    # Compute PDFs
    target_pdf = target_dist.pdf(x)
    proposal_pdf = proposal_dist.pdf(x)
    
    # Create figure
    plt.figure(figsize=(10, 6))
    
    # Plot distributions
    plt.plot(x, target_pdf, 'b-', label='Conditional Distribution')
    plt.plot(x, proposal_pdf, 'g-', label='Proposal Distribution')
    
    # Add threshold line
    plt.axvline(x=threshold, color='r', linestyle='--', label='Threshold')
    
    # Fill the area we're integrating
    x_fill = x[x > threshold]
    plt.fill_between(x_fill, target_pdf[x > threshold], alpha=0.2, color='b',
                    label='Target Probability')
    
    plt.xlabel('x₂')
    plt.ylabel('Probability Density')
    plt.title(f'Distributions and Threshold')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig('examples/importance-sampling-exp/distributions_visualization.png')
    plt.show()

def visualize_samples(mc_samples: np.ndarray, qmc_samples: np.ndarray, 
                     is_samples: np.ndarray, is_qmc_samples: np.ndarray,
                     target_dist: Gaussian1D, proposal_dist: Gaussian1D, 
                     threshold: float):
    """
    Visualize samples from all methods against the true distribution.
    
    Args:
        mc_samples: Samples from Monte Carlo
        qmc_samples: Samples from Quasi-Monte Carlo
        is_samples: Samples from Importance Sampling
        is_qmc_samples: Samples from Importance Sampling with QMC
        target_dist: Target (conditional) distribution
        proposal_dist: Proposal distribution for IS
        threshold: Threshold value for the probability
    """
    plt.figure(figsize=(20, 5))  # Made wider to accommodate 4 plots
    
    # Create points for the true PDF
    x = np.linspace(target_dist.mean - 4*target_dist.std, 
                    proposal_dist.mean + 4*proposal_dist.std, 1000)
    target_pdf = target_dist.pdf(x)
    
    # Plot settings
    methods = [
        (mc_samples, 'blue', 'Monte Carlo'),
        (qmc_samples, 'orange', 'Quasi-Monte Carlo'),
        (is_samples, 'green', 'Importance Sampling'),
        (is_qmc_samples, 'red', 'IS-QMC')  # Added new method
    ]
    
    for idx, (samples, color, label) in enumerate(methods, 1):
        plt.subplot(1, 4, idx)  # Changed to 4 subplots
        
        # Plot histogram of samples
        plt.hist(samples, bins=50, density=True, alpha=0.5, 
                color=color, label=f'{label} Samples')
        
        # Plot true PDF
        plt.plot(x, target_pdf, 'r--', label='True Distribution')
        
        # Add threshold line
        plt.axvline(x=threshold, color='black', linestyle=':', 
                   label='Threshold')
        
        plt.xlabel('x₂')
        plt.ylabel('Density')
        plt.title(f'{label} Sampling')
        plt.grid(True)
        plt.legend()
    
    plt.tight_layout()
    plt.savefig('examples/importance-sampling-exp/sampling_comparison.png')
    plt.show()

def get_ground_truth(x1_fixed: float, threshold: float) -> float:
    """
    Calculates the true conditional probability P(X2 > threshold | X1 = x1_fixed)
    """
    target_dist = get_conditional_distribution(x1_fixed)
    # Using the standard normal CDF: P(X > threshold) = 1 - Φ((threshold - μ)/σ)
    z_score = (threshold - target_dist.mean) / target_dist.std
    return 1 - norm.cdf(z_score)

if __name__ == "__main__":
    # Parameters
    n_samples = 2**9
    x1_fixed = 1.0
    threshold = 1.5
    
    # Setup distributions
    target_dist = get_conditional_distribution(x1_fixed)
    proposal_dist = get_proposal_distribution(target_dist, threshold)
    
    # Visualize distributions
    visualize_distributions(target_dist, proposal_dist, threshold)
    
    # Calculate ground truth
    true_prob = get_ground_truth(x1_fixed, threshold)
    
    # Run comparison
    mc_results, qmc_results, is_results, is_qmc_results = run_comparison(n_samples, x1_fixed, threshold)
    
    # Print final results
    print(f"\nResults for conditional probability P(X₂ > {threshold} | X₁ = {x1_fixed}):")
    print(f"Ground Truth:        {true_prob:.6f}")
    print(f"Monte Carlo:         {mc_results['convergence_data'].running_means[-1]:.6f}")
    print(f"Quasi-Monte Carlo:   {qmc_results['convergence_data'].running_means[-1]:.6f}")
    print(f"Importance Sampling: {is_results['convergence_data'].running_means[-1]:.6f}")
    print(f"IS-QMC:             {is_qmc_results['convergence_data'].running_means[-1]:.6f}")
    
    # Extract samples for visualization
    mc_samples = mc_results['samples']
    qmc_samples = qmc_results['samples']
    is_samples = is_results['samples']
    is_qmc_samples = is_qmc_results['samples']
    
    # Visualize samples
    visualize_samples(mc_samples, qmc_samples, is_samples, is_qmc_samples,
                     target_dist, proposal_dist, threshold)
    
    # Plot convergence results
    plt.figure(figsize=(10, 6))
    
    methods = [
        (mc_results, 'blue', 'Monte Carlo'),
        (qmc_results, 'orange', 'Quasi-Monte Carlo'),
        (is_results, 'green', 'Importance Sampling'),
        (is_qmc_results, 'red', 'IS-QMC')  # Added new method
    ]
    
    for results, color, label in methods:
        data = results['convergence_data']
        stderr = np.sqrt(data.running_variances / data.sample_indices)
        
        plt.fill_between(
            data.sample_indices,
            data.running_means - stderr,
            data.running_means + stderr,
            alpha=0.2,
            color=color,
            label=f'{label} Confidence'
        )
        
        plt.plot(data.sample_indices, data.running_means,
                color=color, linewidth=2, label=label)
    
    plt.xlabel('Number of Samples')
    plt.ylabel('P(X₂ > threshold | X₁ = x₁_fixed)')
    plt.title(f'Convergence Comparison (X₁={x1_fixed}, threshold={threshold})')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig('examples/importance-sampling-exp/conditional_probability.png')
    plt.show() 