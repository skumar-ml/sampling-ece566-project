import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
from distributions.base import Distribution
from distributions.implementations import GMMDistribution
from samplers.implementations import MonteCarloSampler, SobolSampler, TruncatedMHSampler
from main import run_sampling_experiment
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.mixture import GaussianMixture
from scipy.stats import norm

# Define continuous features
CONTINUOUS_FEATURES = ['age', 'bmi', 'children']

class FixedFeatureSampler:
    """Wrapper sampler that adds fixed feature back to samples."""
    def __init__(self, base_sampler, fixed_feature_idx: int, fixed_value: float):
        self.base_sampler = base_sampler
        self.fixed_feature_idx = fixed_feature_idx
        self.fixed_value = fixed_value
    
    def setup(self, distribution, n_dimensions):
        self.base_sampler.setup(distribution, n_dimensions)
    
    def generate_samples(self, n_samples):
        base_samples = self.base_sampler.generate_samples(n_samples)
        # Add fixed feature back
        return np.insert(base_samples, self.fixed_feature_idx, self.fixed_value, axis=1)

def mh_gmm_importance_sampling(conditional_gmm: GMMDistribution, model, cost_threshold: float,
                             fixed_feature_idx: int, fixed_value: float,
                             n_mh_samples: int = 10000):
    """Performs MH-GMM-IS sampling targeting the optimal distribution."""
    def optimal_log_density(x):
        # Add fixed feature back for prediction
        x_with_fixed = np.insert(x.reshape(1, -1), fixed_feature_idx, fixed_value, axis=1)
        # Log density of target * indicator
        cost = model.predict(x_with_fixed)[0]
        if cost > cost_threshold:
            return conditional_gmm.log_pdf(x)
        return -np.inf
    
    # Setup and run MH sampler with optimal target distribution
    mh_sampler = TruncatedMHSampler(
        step_size=1, 
        burn_in=2000,
        log_target_density=optimal_log_density
    )
    mh_sampler.setup(conditional_gmm, conditional_gmm.n_dimensions)
    mh_samples = mh_sampler.generate_samples(n_mh_samples)
    
    # Fit GMM to valid MH samples
    fit_dist = GaussianMixture(n_components=5).fit(mh_samples)
    
    return fit_dist

def mh_gmm_is_estimator(model, cost_threshold: float, 
                       target_dist: GMMDistribution, proposal_dist: GaussianMixture):
    """Returns an importance sampling estimator using GMM proposal."""
    def estimator(x: np.ndarray) -> np.ndarray:
        model_pred = model.predict(x)
        indicator = (model_pred > cost_threshold).astype(float)
        target_density = target_dist.pdf(x[:,1:])
        proposal_density = np.exp(proposal_dist.score_samples(x[:,1:]))
        weights = indicator * target_density / proposal_density
        return weights
    return estimator

def get_conditional_gmm(gmm: GMMDistribution, fixed_feature_idx: int, fixed_value: float) -> GMMDistribution:
    """
    Creates a conditional GMM by adjusting the means and covariances of each component
    given a fixed feature value.
    """
    n_components = len(gmm.weights)
    n_dims = gmm.n_dimensions
    
    # Initialize arrays for new parameters
    new_weights = np.zeros(n_components)
    new_means = np.zeros((n_components, n_dims - 1))
    new_covs = np.zeros((n_components, n_dims - 1, n_dims - 1))
    
    # For each component
    for k in range(n_components):
        mu = gmm.means[k]
        sigma = gmm.covariances[k]
        
        # Split mean and covariance
        mu_1 = mu[fixed_feature_idx]
        mu_2 = np.delete(mu, fixed_feature_idx)
        
        sigma_11 = sigma[fixed_feature_idx, fixed_feature_idx]
        sigma_12 = np.delete(sigma[fixed_feature_idx, :], fixed_feature_idx)
        sigma_21 = np.delete(sigma[:, fixed_feature_idx], fixed_feature_idx)
        sigma_22 = np.delete(np.delete(sigma, fixed_feature_idx, 0), fixed_feature_idx, 1)
        
        # Compute conditional parameters
        conditional_mean = mu_2 + sigma_21 * (fixed_value - mu_1) / sigma_11
        conditional_cov = sigma_22 - np.outer(sigma_21, sigma_12) / sigma_11
        
        # Update component weight based on likelihood of fixed value
        weight_factor = norm.pdf(fixed_value, mu_1, np.sqrt(sigma_11))
        new_weights[k] = gmm.weights[k] * weight_factor
        
        new_means[k] = conditional_mean
        new_covs[k] = conditional_cov
    
    # Normalize weights
    new_weights /= new_weights.sum()
    
    # Create new GMM with conditional parameters
    conditional_gmm = GMMDistribution(n_components=n_components)
    
    # Create a dummy GMM and fit it to some data to initialize all attributes
    dummy_data = np.random.randn(100, n_dims - 1)
    conditional_gmm.gmm = GaussianMixture(
        n_components=n_components, 
        covariance_type='full'
    ).fit(dummy_data)
    
    # Override the fitted parameters with our conditional parameters
    conditional_gmm.gmm.weights_ = new_weights
    conditional_gmm.gmm.means_ = new_means
    conditional_gmm.gmm.covariances_ = new_covs
    
    # Compute precision matrices and their Cholesky decompositions
    conditional_gmm.gmm.precisions_ = np.array([np.linalg.inv(cov) for cov in new_covs])
    conditional_gmm.gmm.precisions_cholesky_ = np.array([
        np.linalg.cholesky(prec).T for prec in conditional_gmm.gmm.precisions_
    ])
    
    conditional_gmm.n_dimensions = n_dims - 1
    
    return conditional_gmm

def threshold_function(x):
    """Function that returns 1 if predicted cost > threshold, 0 otherwise."""
    costs = model.predict(x)
    return (costs > cost_threshold).astype(float)

class GMMProposalDistribution(Distribution):
    """Wrapper for sklearn GaussianMixture to make it compatible with our sampling framework."""
    def __init__(self, gmm: GaussianMixture):
        self.gmm = gmm
        self.n_dimensions = gmm.means_.shape[1]
    
    def sample(self, n_samples: int) -> np.ndarray:
        return self.gmm.sample(n_samples)[0]
    
    def pdf(self, x: np.ndarray) -> np.ndarray:
        return np.exp(self.gmm.score_samples(x))
    
    def log_pdf(self, x: np.ndarray) -> np.ndarray:
        return self.gmm.score_samples(x)

    def inverse_cdf(self, x: np.ndarray) -> np.ndarray:
        pass

if __name__ == "__main__":
    # Load data
    data = pd.read_csv("https://raw.githubusercontent.com/stedy/Machine-Learning-with-R-datasets/master/insurance.csv")
    
    # Extract continuous features
    X = data[CONTINUOUS_FEATURES]
    y = data['charges']
    
    # Train test split and scale features
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train model
    model = GradientBoostingRegressor(
        n_estimators=200,
        max_depth=5,
        min_samples_split=5,
        learning_rate=0.1,
        random_state=42
    )
    model.fit(X_train_scaled, y_train)
    
    # Calculate and print metrics
    train_preds = model.predict(X_train_scaled)
    test_preds = model.predict(X_test_scaled)
    
    train_r2 = r2_score(y_train, train_preds)
    test_r2 = r2_score(y_test, test_preds)
    test_mae = mean_absolute_error(y_test, test_preds)
    
    print("\nModel Performance:")
    print(f"Train R² Score: {train_r2:.4f}")
    print(f"Test R² Score: {test_r2:.4f}")
    print(f"Test MAE: ${test_mae:,.2f}")
    
    # Fit GMM to continuous features
    gmm = GMMDistribution(n_components=1)
    gmm.fit(X_train_scaled)
    
    # Set condition and threshold
    fixed_feature_idx = 0  # age
    fixed_value = scaler.transform([[23, 0, 0]])[0, 0]  # Scale age=23
    cost_threshold = 20000
    n_samples = 2**13
    
    # Get conditional distribution
    conditional_gmm = get_conditional_gmm(gmm, fixed_feature_idx, fixed_value)
    
    # Create wrapped samplers that add back the fixed feature
    mc_sampler = FixedFeatureSampler(MonteCarloSampler(), fixed_feature_idx, fixed_value)
    qmc_sampler = FixedFeatureSampler(SobolSampler(scramble=True), fixed_feature_idx, fixed_value)
    
    # Monte Carlo
    mc_results = run_sampling_experiment(
        distribution=conditional_gmm,
        target_function=threshold_function,
        sampler=mc_sampler,
        n_samples=n_samples,
        n_dimensions=conditional_gmm.n_dimensions
    )
    
    # Quasi-Monte Carlo
    qmc_results = run_sampling_experiment(
        distribution=conditional_gmm,
        target_function=threshold_function,
        sampler=qmc_sampler,
        n_samples=n_samples,
        n_dimensions=conditional_gmm.n_dimensions
    )
    
    # Get GMM proposal distribution using MH samples
    proposal_gmm = mh_gmm_importance_sampling(
        conditional_gmm=conditional_gmm,
        model=model,
        cost_threshold=cost_threshold,
        fixed_feature_idx=fixed_feature_idx,
        fixed_value=fixed_value,
        n_mh_samples=n_samples
    )
    
    # Create proposal distribution
    proposal_dist = GMMProposalDistribution(proposal_gmm)
    
    # Create IS estimator and sampler
    is_estimator = mh_gmm_is_estimator(model, cost_threshold, conditional_gmm, proposal_gmm)
    is_sampler = FixedFeatureSampler(MonteCarloSampler(), fixed_feature_idx, fixed_value)
    
    # MH-GMM Importance Sampling
    is_results = run_sampling_experiment(
        distribution=proposal_dist,
        target_function=is_estimator,
        sampler=is_sampler,
        n_samples=n_samples,
        n_dimensions=conditional_gmm.n_dimensions
    )
    
    # Print results
    print(f"\nProbability of medical costs > ${cost_threshold:,} given age = 23:")
    print(f"Monte Carlo estimate: {mc_results['expectation']:.4f}")
    print(f"Quasi-Monte Carlo estimate: {qmc_results['expectation']:.4f}")
    print(f"MH-GMM-IS estimate: {is_results['expectation']:.4f}")
    
    # Plot convergence
    plt.figure(figsize=(10, 6))
    
    # Plot convergence for each method
    methods = [
        (mc_results, 'blue', 'Monte Carlo'),
        (qmc_results, 'orange', 'Quasi-Monte Carlo'),
        (is_results, 'green', 'MH-GMM-IS')
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
    plt.ylabel('P(cost > threshold | age = 23)')
    plt.title('Convergence of Conditional Probability Estimates')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig('examples/importance-sampling-exp/medical_cost_conditional_convergence.png')
    plt.show()
