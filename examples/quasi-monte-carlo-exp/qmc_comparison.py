import numpy as np
from distributions.implementations import UniformCube
from samplers.implementations import UnitCubeSampler, SobolSampler
from main import run_sampling_experiment
import matplotlib.pyplot as plt

def l1_norm(x: np.ndarray) -> float:
    return np.sum(x)

def run_comparison(n_dimensions: int, n_samples: int):
    # Setup distribution
    dist = UniformCube(n_dimensions)
    
    # Run MC experiment
    mc_sampler = UnitCubeSampler()
    mc_results = run_sampling_experiment(
        distribution=dist,
        target_function=l1_norm,
        sampler=mc_sampler,
        n_samples=n_samples,
        n_dimensions=n_dimensions
    )
    
    # Run QMC experiment
    qmc_sampler = SobolSampler(scramble=True)
    qmc_results = run_sampling_experiment(
        distribution=dist,
        target_function=l1_norm,
        sampler=qmc_sampler,
        n_samples=n_samples,
        n_dimensions=n_dimensions
    )
    
    # Print results
    true_expectation = n_dimensions / 2
    print(f"\nResults for {n_dimensions} dimensions:")
    print(f"Number of samples: {n_samples}")
    print(f"True expectation: {true_expectation:.6f}")
    print(f"MC  estimate: {mc_results['expectation']:.6f}")
    print(f"QMC estimate: {qmc_results['expectation']:.6f}")
    print(f"MC  variance: {mc_results['variance']:.6f}")
    print(f"QMC variance: {qmc_results['variance']:.6f}")
    
    return mc_results, qmc_results

# Run experiments for different dimensions
dimensions = [2, 5, 10, 20]
n_samples = 2**9  # Using power of 2 for Sobol sequences

plt.figure(figsize=(15, 10))

for i, dim in enumerate(dimensions, 1):
    mc_results, qmc_results = run_comparison(dim, n_samples)
    
    plt.subplot(2, 2, i)
    
    # Plot convergence for MC
    mc_data = mc_results['convergence_data']
    mc_stderr = np.sqrt(mc_data.running_variances / mc_data.sample_indices)
    
    plt.fill_between(
        mc_data.sample_indices,
        mc_data.running_means - mc_stderr,
        mc_data.running_means + mc_stderr,
        alpha=0.2,
        color='blue',
        label='MC Confidence'
    )
    
    plt.plot(mc_data.sample_indices, mc_data.running_means,
             color='blue', linewidth=2, label='Monte Carlo')
    
    # Plot convergence for QMC
    qmc_data = qmc_results['convergence_data']
    qmc_stderr = np.sqrt(qmc_data.running_variances / qmc_data.sample_indices)
    
    plt.fill_between(
        qmc_data.sample_indices,
        qmc_data.running_means - qmc_stderr,
        qmc_data.running_means + qmc_stderr,
        alpha=0.2,
        color='orange',
        label='QMC Confidence'
    )
    
    plt.plot(qmc_data.sample_indices, qmc_data.running_means,
             color='orange', linewidth=2, label='Quasi-Monte Carlo')
    
    plt.axhline(y=dim/2, color='r', linestyle='--', label='True Value')
    plt.xlabel('Number of Samples')
    plt.ylabel('Estimate')
    plt.title(f'Convergence ({dim} dimensions)')
    plt.grid(True)
    plt.legend()

plt.tight_layout()
plt.show()
plt.savefig('examples/quasi-monte-carlo-exp/qmc_comparison.png')
# Visualize 2D point sets for comparison
if 2 in dimensions:
    n_vis_samples = 1000
    
    # Generate 2D samples
    dist = UniformCube(2)
    
    mc_sampler = UnitCubeSampler()
    mc_sampler.setup(dist, 2)
    mc_points = mc_sampler.generate_samples(n_vis_samples)
    
    qmc_sampler = SobolSampler(scramble=True)
    qmc_sampler.setup(dist, 2)
    qmc_points = qmc_sampler.generate_samples(n_vis_samples)
    
    # Plot points
    plt.figure(figsize=(12, 5))
    
    plt.subplot(121)
    plt.scatter(mc_points[:, 0], mc_points[:, 1], alpha=0.5, s=10)
    plt.title('Monte Carlo Samples')
    plt.xlabel('x₁')
    plt.ylabel('x₂')
    plt.grid(True)
    
    plt.subplot(122)
    plt.scatter(qmc_points[:, 0], qmc_points[:, 1], alpha=0.5, s=10)
    plt.title('Quasi-Monte Carlo (Sobol) Samples')
    plt.xlabel('x₁')
    plt.ylabel('x₂')
    plt.grid(True)
    
    plt.tight_layout()
    plt.show() 
plt.savefig('examples/quasi-monte-carlo-exp/qmc_comparison_2d.png')