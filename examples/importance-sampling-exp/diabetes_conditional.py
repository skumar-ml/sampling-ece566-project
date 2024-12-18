import numpy as np
from distributions.implementations import Gaussian1D
from samplers.implementations import UnitCubeSampler, SobolSampler
from main import run_sampling_experiment
import matplotlib.pyplot as plt
from scipy.stats import norm
from sklearn.datasets import load_diabetes
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Load the diabetes dataset
data = load_diabetes()
X, y = data['data'], data['target']

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split into train/test for simplicity
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Define conditional distribution for diabetes dataset
def get_conditional_distribution_diabetes(fixed_value: float, feature_index: int):
    """
    Creates a conditional distribution for the diabetes dataset.
    Args:
        fixed_value: Fixed value of the feature
        feature_index: Index of the feature being conditioned on
    """
    relevant_feature = X_train[:, feature_index]
    target_conditioned = y_train[relevant_feature > fixed_value]
    conditional_mean = target_conditioned.mean() if len(target_conditioned) > 0 else y_train.mean()
    conditional_std = target_conditioned.std() if len(target_conditioned) > 0 else y_train.std()
    return Gaussian1D(mean=conditional_mean, std=conditional_std)

# Get conditional and proposal distributions
feature_index = 2  # Example: third feature
fixed_feature_value = X_train[:, feature_index].mean()
target_threshold = y_train.mean() + y_train.std()

conditional_dist = get_conditional_distribution_diabetes(fixed_feature_value, feature_index)
proposal_dist = Gaussian1D(mean=target_threshold, std=conditional_dist.std)

# Define estimators
def conditional_probability_estimator(threshold: float):
    def estimator(x: np.ndarray) -> float:
        return (x > threshold).astype(float)
    return estimator

def importance_sampling_estimator(threshold: float, target_dist: Gaussian1D, proposal_dist: Gaussian1D):
    def estimator(x: np.ndarray) -> float:
        indicator = (x > threshold).astype(float)
        weights = target_dist.pdf(x) / proposal_dist.pdf(x)
        return indicator * weights
    return estimator

# Run experiments
n_samples = 2**9

mc_sampler = UnitCubeSampler()
qmc_sampler = SobolSampler(scramble=True)

mc_estimator = conditional_probability_estimator(target_threshold)
qmc_estimator = conditional_probability_estimator(target_threshold)
is_estimator = importance_sampling_estimator(target_threshold, conditional_dist, proposal_dist)
is_qmc_estimator = importance_sampling_estimator(target_threshold, conditional_dist, proposal_dist)

mc_results = run_sampling_experiment(
    distribution=conditional_dist,
    target_function=mc_estimator,
    sampler=mc_sampler,
    n_samples=n_samples,
    n_dimensions=1
)

qmc_results = run_sampling_experiment(
    distribution=conditional_dist,
    target_function=qmc_estimator,
    sampler=qmc_sampler,
    n_samples=n_samples,
    n_dimensions=1
)

is_results = run_sampling_experiment(
    distribution=proposal_dist,
    target_function=is_estimator,
    sampler=mc_sampler,
    n_samples=n_samples,
    n_dimensions=1
)

is_qmc_results = run_sampling_experiment(
    distribution=proposal_dist,
    target_function=is_qmc_estimator,
    sampler=qmc_sampler,
    n_samples=n_samples,
    n_dimensions=1
)

# Visualize distributions
def visualize_distributions(target_dist, proposal_dist, threshold):
    x = np.linspace(target_dist.mean - 4*target_dist.std, 
                    proposal_dist.mean + 4*proposal_dist.std, 1000)
    plt.figure(figsize=(10, 6))
    plt.plot(x, target_dist.pdf(x), label='Target Distribution')
    plt.plot(x, proposal_dist.pdf(x), label='Proposal Distribution')
    plt.axvline(threshold, linestyle='--', color='r', label='Threshold')
    plt.legend()
    plt.grid()
    plt.show()

visualize_distributions(conditional_dist, proposal_dist, target_threshold)

# Print results
print(f"Results for P(Target > {target_threshold} | Feature[{feature_index}] = {fixed_feature_value}):")
print(f"Monte Carlo: {mc_results['convergence_data'].running_means[-1]:.6f}")
print(f"Quasi-Monte Carlo: {qmc_results['convergence_data'].running_means[-1]:.6f}")
print(f"Importance Sampling: {is_results['convergence_data'].running_means[-1]:.6f}")
print(f"IS-QMC: {is_qmc_results['convergence_data'].running_means[-1]:.6f}")

# Visualize samples and convergence
def visualize_samples(samples_dict, target_dist, proposal_dist, threshold):
    x = np.linspace(target_dist.mean - 4*target_dist.std, 
                    proposal_dist.mean + 4*proposal_dist.std, 1000)
    plt.figure(figsize=(15, 6))
    for i, (key, samples) in enumerate(samples_dict.items(), 1):
        plt.subplot(1, 4, i)
        plt.hist(samples, bins=50, density=True, alpha=0.6, label=f'{key} Samples')
        plt.plot(x, target_dist.pdf(x), 'r--', label='True Distribution')
        plt.axvline(threshold, linestyle='--', color='black')
        plt.legend()
        plt.grid()
    plt.show()

visualize_samples(
    {
        "Monte Carlo": mc_results['samples'],
        "Quasi-Monte Carlo": qmc_results['samples'],
        "Importance Sampling": is_results['samples'],
        "IS-QMC": is_qmc_results['samples']
    },
    conditional_dist, proposal_dist, target_threshold
)


# Plot convergence results
plt.figure(figsize=(10, 6))

# Add ground truth line (if calculable)
# Here, we use a simple approximation; adapt if you can compute true P(target > threshold | feature = fixed_value)
true_prob = 1 - norm.cdf((target_threshold - conditional_dist.mean) / conditional_dist.std)
plt.axhline(y=true_prob, color='black', linestyle=':', linewidth=2, label='Ground Truth')

# Convergence data for different methods
methods = [
    (mc_results, 'blue', 'Monte Carlo'),
    (qmc_results, 'orange', 'Quasi-Monte Carlo'),
    (is_results, 'green', 'Importance Sampling'),
    (is_qmc_results, 'red', 'IS-QMC')
]

for results, color, label in methods:
    data = results['convergence_data']
    stderr = np.sqrt(data.running_variances / data.sample_indices)
    
    # Confidence intervals
    plt.fill_between(
        data.sample_indices,
        data.running_means - stderr,
        data.running_means + stderr,
        alpha=0.2,
        color=color,
        label=f'{label} Confidence'
    )
    
    # Running means
    plt.plot(data.sample_indices, data.running_means, color=color, linewidth=2, label=label)

plt.xlabel('Number of Samples')
plt.ylabel(f'P(Target > {target_threshold:.2f} | Feature[{feature_index}] = {fixed_feature_value:.2f})')
plt.title(f'Convergence of Conditional Probability Estimation')
plt.grid(True)
plt.legend()
plt.tight_layout()

# Save the plot
output_path = 'conditional_probability.png'
plt.savefig(output_path)
print(f"Saved convergence plot as {output_path}")
plt.show()
