import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
from distributions.implementations import GMMDistribution
from samplers.implementations import MonteCarloSampler, SobolSampler, TruncatedMHSampler
from main import run_sampling_experiment
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_absolute_error
from scipy.stats import norm, gaussian_kde
from sklearn.mixture import GaussianMixture
from typing import List, Tuple

# At the top of the file, add these constants
CONTINUOUS_FEATURES = ['age', 'bmi', 'children']
CATEGORICAL_FEATURES = ['sex', 'smoker', 'region']

def prepare_features(data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, List[dict]]:
    """
    Prepare features by separating continuous and categorical variables.
    Returns:
        - continuous features DataFrame
        - full encoded features DataFrame
        - list of categorical feature mappings
    """
    df = data.copy()
    
    # Get all unique combinations of categorical variables
    categorical_combinations = df[CATEGORICAL_FEATURES].drop_duplicates()
    categorical_mappings = categorical_combinations.to_dict('records')
    
    # One-hot encode categorical variables for the full dataset
    df_encoded = pd.get_dummies(df, columns=CATEGORICAL_FEATURES)
    
    return df[CONTINUOUS_FEATURES], df_encoded, categorical_mappings

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

def get_truncation_threshold(model, conditional_gmm: GMMDistribution, cost_threshold: float,
                           fixed_feature_idx: int, fixed_value: float,
                           n_samples: int = 10000) -> np.ndarray:
    """
    Find reasonable truncation thresholds by sampling from conditional GMM
    and identifying regions where cost > threshold.
    """
    samples = conditional_gmm.sample(n_samples)
    samples_full = np.insert(samples, fixed_feature_idx, fixed_value, axis=1)
    costs = model.predict(samples_full)
    high_cost_samples = samples[costs > cost_threshold]
    
    # Use minimum values as thresholds with a small buffer
    thresholds = np.min(high_cost_samples, axis=0) - 0.1
    return thresholds

def mh_gmm_importance_sampling(conditional_gmm: GMMDistribution, model, cost_threshold: float,
                             fixed_feature_idx: int, fixed_value: float,
                             categorical_mappings: List[dict], scaler: StandardScaler,
                             feature_names: List[str], n_mh_samples: int = 10000):
    """Performs MH-KDE-IS sampling targeting the optimal distribution."""
    def optimal_log_density(x):
        # Add fixed feature back
        x_with_fixed = np.insert(x.reshape(1, -1), fixed_feature_idx, fixed_value, axis=1)
        
        # Create all categorical permutations in scaled space
        x_full = marginalize_samples(
            x_with_fixed, 
            categorical_mappings,
            scaler,
            feature_names
        )
        
        # Compute costs for all permutations
        costs = model.predict(x_full)
        
        # If no permutation exceeds threshold, return -inf
        if not np.any(costs > cost_threshold):
            return -np.inf
        
        # Return log of target density times average indicator
        log_target = conditional_gmm.log_pdf(x)
        indicator_mean = np.mean(costs > cost_threshold)
        
        return log_target + np.log(indicator_mean)
    
    # Setup and run MH sampler with optimal target distribution
    mh_sampler = TruncatedMHSampler(
        step_size=0.01, 
        burn_in=2000,
        log_target_density=optimal_log_density
    )
    mh_sampler.setup(conditional_gmm, conditional_gmm.n_dimensions)
    mh_samples = mh_sampler.generate_samples(n_mh_samples)
    
    # Fit gmm to MH samples
    fit_dist = GaussianMixture(n_components=10).fit(mh_samples)
    
    return fit_dist

def mh_gmm_is_estimator(model, cost_threshold: float, 
                       target_dist: GMMDistribution, proposal_dist: GaussianMixture):
    """Returns an importance sampling estimator using KDE proposal."""
    def estimator(x: np.ndarray) -> np.ndarray:
        model_pred = model.predict(x)
        indicator = (model_pred > cost_threshold).astype(float)

        target_density = target_dist.pdf(x[:, 1:3]) # Ignores age and categorical features
        proposal_density = np.exp(proposal_dist.score_samples(x[:, 1:3]))
        weights = indicator * target_density / proposal_density
        return weights
    return estimator

def marginalize_samples(samples: np.ndarray, categorical_mappings: List[dict], 
                       scaler: StandardScaler, feature_names: List[str]) -> np.ndarray:
    """
    Duplicate samples for each categorical combination and add encoded categorical features.
    
    Args:
        samples: Array of continuous feature samples (already scaled)
        categorical_mappings: List of dictionaries containing categorical combinations
        scaler: StandardScaler used for the full feature set
        feature_names: List of all feature names in correct order
    """
    n_samples = len(samples)
    n_combinations = len(categorical_mappings)
    
    # Create dummy DataFrame with categorical combinations
    dummy_df = pd.DataFrame(categorical_mappings)
    
    # Add dummy continuous features (they'll be replaced later)
    for feat in CONTINUOUS_FEATURES:
        dummy_df[feat] = 0.0
    
    # One-hot encode categorical features
    encoded_categorical = pd.get_dummies(dummy_df, columns=CATEGORICAL_FEATURES)
    
    # Ensure all expected columns are present
    for col in feature_names:
        if col not in encoded_categorical.columns:
            encoded_categorical[col] = 0
    
    # Reorder columns to match feature_names
    encoded_categorical = encoded_categorical[feature_names]
    
    # Get the transformed categorical features
    categorical_transformed = scaler.transform(encoded_categorical)
    
    # Initialize output array
    # Shape will be (n_samples * n_combinations, n_features)
    output = np.zeros((n_samples * n_combinations, len(feature_names)))
    
    # For each sample, create versions with all categorical combinations
    for i in range(n_samples):
        start_idx = i * n_combinations
        end_idx = start_idx + n_combinations
        
        # Copy the categorical features for this sample
        output[start_idx:end_idx] = categorical_transformed
        
        # Replace the continuous features with the actual sample values
        for j, feat in enumerate(CONTINUOUS_FEATURES):
            feat_idx = list(feature_names).index(feat)
            output[start_idx:end_idx, feat_idx] = samples[i, j]
    
    return output

class MarginalizationMixin:
    """Mixin class that handles post-sampling marginalization over categorical variables."""
    
    def __init__(self, fixed_feature_idx: int, fixed_value: float,
                 categorical_mappings: List[dict], scaler: StandardScaler,
                 feature_names: List[str]):
        self.fixed_feature_idx = fixed_feature_idx
        self.fixed_value = fixed_value
        self.categorical_mappings = categorical_mappings
        self.scaler = scaler
        self.feature_names = feature_names
    
    def process_samples(self, samples: np.ndarray) -> np.ndarray:
        """Add fixed feature and marginalize over categorical variables."""
        # First add the fixed feature back
        samples_with_fixed = np.insert(samples, self.fixed_feature_idx, 
                                     self.fixed_value, axis=1)
        
        # Then marginalize over categorical variables
        return marginalize_samples(samples_with_fixed, self.categorical_mappings,
                                 self.scaler, self.feature_names)

class MarginalizedMonteCarloSampler(MarginalizationMixin, MonteCarloSampler):
    """Monte Carlo sampler with categorical marginalization."""
    
    def __init__(self, fixed_feature_idx: int, fixed_value: float,
                 categorical_mappings: List[dict], scaler: StandardScaler,
                 feature_names: List[str]):
        MarginalizationMixin.__init__(self, fixed_feature_idx, fixed_value,
                                    categorical_mappings, scaler, feature_names)
        MonteCarloSampler.__init__(self)
    
    def generate_samples(self, n_samples: int) -> np.ndarray:
        # Get base samples from parent class
        base_samples = super().generate_samples(n_samples)
        # Process samples through marginalization
        return self.process_samples(base_samples)

class MarginalizedSobolSampler(MarginalizationMixin, SobolSampler):
    """Sobol sampler with categorical marginalization."""
    
    def __init__(self, fixed_feature_idx: int, fixed_value: float,
                 categorical_mappings: List[dict], scaler: StandardScaler,
                 feature_names: List[str], scramble: bool = True):
        MarginalizationMixin.__init__(self, fixed_feature_idx, fixed_value,
                                    categorical_mappings, scaler, feature_names)
        SobolSampler.__init__(self, scramble=scramble)
    
    def generate_samples(self, n_samples: int) -> np.ndarray:
        # Get base samples from parent class
        base_samples = super().generate_samples(n_samples)
        # Process samples through marginalization
        return self.process_samples(base_samples)

class MarginalizedGMMSampler(MarginalizationMixin, MonteCarloSampler):
    """GMM sampler with categorical marginalization."""
    
    def __init__(self, gmm: GaussianMixture,
                 fixed_feature_idx: int, fixed_value: float,
                 categorical_mappings: List[dict], scaler: StandardScaler,
                 feature_names: List[str]):
        MarginalizationMixin.__init__(self, fixed_feature_idx, fixed_value,
                                    categorical_mappings, scaler, feature_names)
        MonteCarloSampler.__init__(self)
        self.gmm = gmm
    
    def generate_samples(self, n_samples: int) -> np.ndarray:
        # Sample from GMM
        gmm_samples = self.gmm.sample(n_samples)[0]
        # Process samples through marginalization
        return self.process_samples(gmm_samples)

def create_threshold_function(model, threshold: float):
    """Creates a function that returns 1 if predicted cost > threshold, 0 otherwise."""
    def threshold_function(x: np.ndarray) -> np.ndarray:
        costs = model.predict(x)
        return (costs > threshold).astype(float)
    return threshold_function

if __name__ == "__main__":
    # Load data
    data = pd.read_csv("https://raw.githubusercontent.com/stedy/Machine-Learning-with-R-datasets/master/insurance.csv")
    
    # Prepare features
    X_continuous, X_encoded, categorical_mappings = prepare_features(data)
    X = X_encoded.drop(columns=['charges'])  # Full encoded features for model training
    y = data['charges']
    
    # Train test split and scale continuous features
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
    
    # Get indices of continuous features in the encoded dataset
    continuous_feature_indices = [list(X.columns).index(feat) for feat in CONTINUOUS_FEATURES]
    
    # Fit GMM only to continuous features using the indices
    gmm = GMMDistribution(n_components=10)
    gmm.fit(X_train_scaled[:, continuous_feature_indices])
    
    # Set condition and threshold
    fixed_feature_idx = 0  # age
    
    # Create a dummy row with mean values for all features except age
    dummy_row = pd.DataFrame(columns=X.columns)
    dummy_row.loc[0] = X_train.mean()  # Fill with mean values
    dummy_row.iloc[0, fixed_feature_idx] = 23  # Set age to 23
    
    # Scale the entire feature vector
    fixed_value = scaler.transform(dummy_row)[0, fixed_feature_idx]
    cost_threshold = 20000
    n_samples = 2**9
    
    # Get conditional distribution
    conditional_gmm = get_conditional_gmm(gmm, fixed_feature_idx, fixed_value)
    
    # Run experiments with new samplers
    mc_sampler = MarginalizedMonteCarloSampler(
        fixed_feature_idx=fixed_feature_idx,
        fixed_value=fixed_value,
        categorical_mappings=categorical_mappings,
        scaler=scaler,
        feature_names=X.columns
    )
    
    qmc_sampler = MarginalizedSobolSampler(
        fixed_feature_idx=fixed_feature_idx,
        fixed_value=fixed_value,
        categorical_mappings=categorical_mappings,
        scaler=scaler,
        feature_names=X.columns,
        scramble=True
    )
    
    # Create threshold function
    threshold_function = create_threshold_function(model, cost_threshold)
    
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
    
    # Get KDE proposal distribution using MH samples targeting optimal distribution
    gmm = mh_gmm_importance_sampling(
        conditional_gmm=conditional_gmm,
        model=model,
        cost_threshold=cost_threshold,
        n_mh_samples=n_samples,
        fixed_feature_idx=fixed_feature_idx,
        fixed_value=fixed_value,
        categorical_mappings=categorical_mappings,
        scaler=scaler,
        feature_names=X.columns
    )
    
    # Create marginalized KDE sampler
    gmm_sampler = MarginalizedGMMSampler(
        gmm=gmm,
        fixed_feature_idx=fixed_feature_idx,
        fixed_value=fixed_value,
        categorical_mappings=categorical_mappings,
        scaler=scaler,
        feature_names=X.columns
    )
    
    # Create IS estimator
    is_estimator = mh_gmm_is_estimator(model, cost_threshold, conditional_gmm, gmm)
    
    # MH-KDE Importance Sampling
    is_results = run_sampling_experiment(
        distribution=None,  # Not needed as we use kde_samples directly
        target_function=is_estimator,
        sampler=gmm_sampler,
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
    
    # Plot MC convergence
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
    
    # Plot QMC convergence
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
    
    # Add MH-KDE-IS to the convergence plot
    is_data = is_results['convergence_data']
    is_stderr = np.sqrt(is_data.running_variances / is_data.sample_indices)
    
    plt.fill_between(
        is_data.sample_indices,
        is_data.running_means - is_stderr,
        is_data.running_means + is_stderr,
        alpha=0.2,
        color='green',
        label='MH-GMM-IS Confidence'
    )
    plt.plot(is_data.sample_indices, is_data.running_means,
             color='green', linewidth=2, label='MH-GMM-IS')
    
    plt.xlabel('Number of Samples')
    plt.ylabel('P(cost > threshold | age = 23)')
    plt.title('Convergence of Conditional Probability Estimates')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig('examples/importance-sampling-exp/medical_cost_conditional_convergence.png')
    plt.show()
