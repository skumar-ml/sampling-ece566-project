import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from distributions.implementations import GMMDistribution
from samplers.implementations import MonteCarloSampler, SobolSampler
from main import run_sampling_experiment
import matplotlib.pyplot as plt


def load_data(file_path: str) -> pd.DataFrame:
    """
    Loads medical cost data from a CSV file.
    Args:
        file_path (str): Path to the CSV file.
    Returns:
        pd.DataFrame: DataFrame containing the dataset.
    """
    try:
        data = pd.read_csv(file_path)
        if 'charges' not in data.columns:
            raise ValueError("'charges' column not found in the dataset.")
        return data
    except Exception as e:
        print(f"Error loading data: {e}")
        exit(1)


def prepare_features(data: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare features with one-hot encoding for categorical variables.
    """
    # Create a copy to avoid modifying the original
    df = data.copy()
    
    # One-hot encode categorical variables
    df = pd.get_dummies(df, columns=['sex', 'region', 'smoker'])
    
    return df


def train_model(data: pd.DataFrame) -> tuple:
    """
    Trains a gradient boosting model to predict medical costs.
    Returns the model, scaler, and feature names.
    """
    # Prepare features
    df = prepare_features(data)
    
    # Separate features and target
    X = df.drop(columns=['charges'])
    y = df['charges']
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train the model
    model = GradientBoostingRegressor(
        n_estimators=200,
        max_depth=5,
        min_samples_split=5,
        learning_rate=0.1,
        random_state=42
    )
    
    # Fit the model
    model.fit(X_train_scaled, y_train)
    
    # Make predictions
    train_preds = model.predict(X_train_scaled)
    test_preds = model.predict(X_test_scaled)
    
    # Calculate and print metrics
    train_r2 = r2_score(y_train, train_preds)
    test_r2 = r2_score(y_test, test_preds)
    test_mae = mean_absolute_error(y_test, test_preds)
    
    print("\nModel Performance:")
    print(f"Train R² Score: {train_r2:.4f}")
    print(f"Test R² Score: {test_r2:.4f}")
    print(f"Test MAE: ${test_mae:,.2f}")
    
    return model, scaler, X.columns


def run_comparison(distribution: GMMDistribution, model: GradientBoostingRegressor, n_samples: int):
    """Run comparison between MC and QMC sampling."""
    
    # Run MC experiment
    mc_sampler = MonteCarloSampler()
    mc_results = run_sampling_experiment(
        distribution=distribution,
        target_function=model.predict,
        sampler=mc_sampler,
        n_samples=n_samples,
        n_dimensions=distribution.n_dimensions
    )
    
    # Run QMC experiment
    qmc_sampler = SobolSampler(scramble=True)
    qmc_results = run_sampling_experiment(
        distribution=distribution,
        target_function=model.predict,
        sampler=qmc_sampler,
        n_samples=n_samples,
        n_dimensions=distribution.n_dimensions
    )
    
    return mc_results, qmc_results


if __name__ == "__main__":
    # Load and prepare data
    file_path = "https://raw.githubusercontent.com/stedy/Machine-Learning-with-R-datasets/master/insurance.csv"
    medical_cost_data = load_data(file_path)
    
    # Train model and get scaler
    model, scaler, feature_names = train_model(medical_cost_data)
    
    # Create and scale features for GMM fitting
    df_prepared = prepare_features(medical_cost_data)
    X_scaled = scaler.transform(df_prepared[feature_names])
    
    # Setup distribution
    distribution = GMMDistribution(n_components=3)
    distribution.fit(X_scaled, feature_names)
    
    # Run experiments
    n_samples = 2**11  # Using power of 2 for Sobol sequences
    mc_results, qmc_results = run_comparison(distribution, model, n_samples)
    
    # Print results
    true_mean = medical_cost_data['charges'].mean()
    print("\nMedical Cost Estimation Results:")
    print(f"Number of samples: {n_samples}")
    print(f"Training Data Mean: ${true_mean:,.2f}")
    print(f"MC  estimate: ${mc_results['expectation']:,.2f}")
    print(f"QMC estimate: ${qmc_results['expectation']:,.2f}")
    print(f"MC  variance: ${mc_results['variance']:,.2f}")
    print(f"QMC variance: ${qmc_results['variance']:,.2f}")
    
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
    
    plt.axhline(y=true_mean, color='r', linestyle='--', label='Training Data Mean')
    plt.xlabel('Number of Samples')
    plt.ylabel('Estimated Mean Cost ($)')
    plt.title('Convergence of Cost Estimates')
    plt.grid(True)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('examples/quasi-monte-carlo-exp/med_cost_images/convergence.png')
    plt.show()

