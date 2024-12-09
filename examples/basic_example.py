import numpy as np
from distributions.implementations import MultivariateNormal
from samplers.implementations import MetropolisHastingsSampler
from main import run_sampling_experiment

# Define target function (e.g., compute mean of first dimension)
def target_function(x: np.ndarray) -> float:
    return x[0]

# Setup distribution
dist = MultivariateNormal()
dist.setup(n_dimensions=5)

# Setup sampler
sampler = MetropolisHastingsSampler(step_size=0.1, burn_in=1000)

# Run experiment
results = run_sampling_experiment(
    distribution=dist,
    target_function=target_function,
    sampler=sampler,
    n_samples=10000,
    n_dimensions=5
)

print(f"Estimated expectation: {results['expectation']:.4f}")
print(f"Estimated variance: {results['variance']:.4f}") 