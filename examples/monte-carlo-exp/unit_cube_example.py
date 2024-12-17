import numpy as np
from distributions.implementations import UniformCube
from samplers.implementations import UnitCubeSampler
from main import run_sampling_experiment
import matplotlib.pyplot as plt

# Define target function (average of coordinates)
def average_coordinate(x: np.ndarray) -> float:
    return np.mean(x)

# Parameters
n_dimensions = 5
n_samples = 10000

# Setup distribution
dist = UniformCube(n_dimensions)

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

# Calculate standard errors for confidence bands
standard_errors = np.sqrt(conv_data.running_variances / conv_data.sample_indices)

# Create confidence bands
upper_bound = conv_data.running_means + standard_errors
lower_bound = conv_data.running_means - standard_errors

# Plot shaded confidence region
plt.fill_between(
    conv_data.sample_indices, 
    lower_bound, 
    upper_bound, 
    alpha=0.2,  # Transparent shading
    color='blue'  # Match line color
)

# Plot main line on top
plt.plot(conv_data.sample_indices, conv_data.running_means, 
         color='blue', 
         linewidth=2, 
         label='Running Mean')

plt.axhline(y=0.5, color='r', linestyle='--', label='True Mean')
plt.xlabel('Number of Samples')
plt.ylabel('Estimate')
plt.title('Convergence of Monte Carlo Estimate')
plt.legend()
plt.grid(True)
plt.show()
plt.savefig('examples/monte-carlo-exp/unit_cube_example.png')