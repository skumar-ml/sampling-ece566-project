import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


# Load medical cost data
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


# Monte Carlo sampling experiment
def run_experiment(data: np.ndarray, n_samples: int):
    """
    Runs a Monte Carlo experiment to estimate the mean and variance.
    Args:
        data (np.ndarray): Array of values (e.g., costs or predictions).
        n_samples (int): Number of Monte Carlo samples.
    Returns:
        dict: Contains results of the experiment.
    """
    np.random.seed(42)  # For reproducibility
    sample_indices = np.random.randint(0, len(data), size=n_samples)
    samples = data[sample_indices]

    # Compute running sums, means, and variances
    running_sums = np.cumsum(samples)
    running_sums_sq = np.cumsum(samples ** 2)
    running_means = running_sums / np.arange(1, n_samples + 1)
    running_variances = running_sums_sq / np.arange(1, n_samples + 1) - running_means ** 2

    return {
        "expectation": running_means[-1],
        "variance": running_variances[-1],
        "convergence_data": {
            "running_means": running_means,
            "running_variances": running_variances,
            "sample_indices": np.arange(1, n_samples + 1)
        }
    }


# Plot convergence
def plot_convergence(results: dict, true_mean: float, file_name: str = None):
    """
    Plots the convergence of the Monte Carlo estimate.
    Args:
        results (dict): Results dictionary containing running means and variances.
        true_mean (float): True mean of the values.
        file_name (str): File name to save the plot (optional).
    """
    convergence_data = results['convergence_data']
    running_means = convergence_data['running_means']
    sample_indices = convergence_data['sample_indices']
    running_variances = convergence_data['running_variances']

    # Calculate standard errors
    standard_errors = np.sqrt(running_variances / sample_indices)

    # Confidence bands
    upper_bound = running_means + standard_errors
    lower_bound = running_means - standard_errors

    plt.figure(figsize=(10, 6))

    # Plot running means with confidence bands
    plt.fill_between(
        sample_indices,
        lower_bound,
        upper_bound,
        alpha=0.2,
        color='blue',
        label='Confidence Band'
    )
    plt.plot(sample_indices, running_means, color='blue', linewidth=2, label='Running Mean')
    plt.axhline(y=true_mean, color='red', linestyle='--', label='True Mean')

    plt.xlabel('Number of Samples')
    plt.ylabel('Estimate of Mean')
    plt.title('Convergence of Monte Carlo Estimate')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    if file_name:
        plt.savefig(file_name)
    plt.show()


# Train a linear regression model
def train_model(data: pd.DataFrame) -> np.ndarray:
    """
    Trains a linear regression model to predict medical costs.
    Args:
        data (pd.DataFrame): DataFrame containing the dataset.
    Returns:
        np.ndarray: Predicted medical costs.
    """
    # Prepare features and target
    features = data.drop(columns=['charges'])
    target = data['charges']

    # One-hot encoding for categorical variables
    features = pd.get_dummies(features, drop_first=True)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

    # Train the model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Evaluate the model
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    print(f"Model Mean Squared Error: {mse:.6f}")

    return model.predict(features)


if __name__ == "__main__":
    # Load data
    file_path = "https://raw.githubusercontent.com/stedy/Machine-Learning-with-R-datasets/master/insurance.csv"
    medical_cost_data = load_data(file_path)

    # Parameters
    n_samples = 512
    # Estimate true mean of medical costs
    true_mean = medical_cost_data['charges'].mean()

    # Run Monte Carlo experiment for medical cost
    cost_results = run_experiment(medical_cost_data['charges'].values, n_samples)

    # Print results for medical cost
    print("=== Medical Cost Estimation ===")
    print(f"Number of samples: {n_samples}")
    print(f"Estimated expectation: {cost_results['expectation']:.6f}")
    print(f"True expectation: {true_mean:.6f}")
    print(f"Estimated variance: {cost_results['variance']:.6f}")

    # Plot convergence for medical cost
    plot_convergence(cost_results, true_mean, file_name='examples/monte-carlo-exp/medical_cost_convergence.png')

    # Train a model to predict medical costs
    model_predictions = train_model(medical_cost_data)

    # Estimate true mean of model predictions
    true_model_mean = model_predictions.mean()

    # Run Monte Carlo experiment for model predictions
    model_results = run_experiment(model_predictions, n_samples)

    # Print results for model predictions
    print("\n=== Model Prediction Estimation ===")
    print(f"Estimated model expectation: {model_results['expectation']:.6f}")
    print(f"True model expectation: {true_model_mean:.6f}")
    print(f"Estimated model variance: {model_results['variance']:.6f}")

    # Plot convergence for model predictions
    plot_convergence(
        model_results,
        true_model_mean,
        file_name='examples/monte-carlo-exp/model_prediction_convergence.png'
    )
