import numpy as np
from distributions.implementations import UniformCube
from samplers.implementations import UniformSampler
from main import run_sampling_experiment
import matplotlib.pyplot as plt

# Define target function (average of coordinates)
def average_coordinate(x: np.ndarray) -> float:
    return np.mean(x)

# TODO: Repeat experiment as n_dimensions increases

# Parameters
n_dimensions = 5
n_samples = 10000

# Setup distribution
dist = UniformCube(n_dimensions)

# Modify UniformSampler for [0,1] range instead of [-1,1]
class UnitCubeSampler(UniformSampler):
    def generate_samples(self, n_samples: int) -> np.ndarray:
        return np.random.uniform(
            low=0, high=1, 
            size=(n_samples, self.n_dimensions)
        )

# Setup sampler
sampler = UnitCubeSampler()

# Run experiment
results = run_sampling_experiment(
    distribution=dist,
    target_function=average_coordinate,
    sampler=sampler,
    n_samples=n_samples,
    n_dimensions=n_dimensions
)

# Print results
print(f"Number of dimensions: {n_dimensions}")
print(f"Number of samples: {n_samples}")
print(f"Estimated expectation: {results['expectation']:.6f}")
print(f"True expectation: 0.500000")  # The true mean is 0.5
print(f"Estimated variance: {results['variance']:.6f}")

# Plot convergence
conv_data = results['convergence_data']
plt.figure(figsize=(10, 6))
plt.plot(conv_data.sample_indices, conv_data.running_means, label='Running Mean')
plt.axhline(y=0.5, color='r', linestyle='--', label='True Mean')
plt.xlabel('Number of Samples')
plt.ylabel('Estimate')
plt.title('Convergence of Monte Carlo Estimate')
plt.legend()
plt.grid(True)
plt.show()

# Plot error
plt.figure(figsize=(10, 6))
plt.plot(conv_data.sample_indices, np.abs(conv_data.running_means - 0.5), label='Absolute Error')
plt.xlabel('Number of Samples')
plt.ylabel('Absolute Error')
plt.title('Error Convergence')
plt.yscale('log')
plt.grid(True)
plt.legend()
plt.show() 