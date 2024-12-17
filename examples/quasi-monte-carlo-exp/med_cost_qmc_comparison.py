import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from distributions.implementations import UniformCube
from samplers.implementations import UnitCubeSampler, SobolSampler
from main import run_sampling_experiment
import matplotlib.pyplot as plt

# Load the medical cost dataset
medical_cost_df = pd.read_csv("https://raw.githubusercontent.com/stedy/Machine-Learning-with-R-datasets/master/insurance.csv")

# One-hot encode categorical features
medical_cost_df = pd.get_dummies(medical_cost_df, drop_first=True)

# Split into features and target
X = medical_cost_df.drop(columns=['charges'])  # 'charges' is the target
y = medical_cost_df['charges']

# Normalize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split into train-test sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train a Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"Linear Regression Test MSE: {mse:.6f}")

# Function to predict costs based on input samples
def model_prediction(x: np.ndarray) -> float:
    return model.predict([x])[0]

# Define a function to run MC and QMC experiments
def run_comparison_medical(n_dimensions: int, n_samples: int):
    # Setup uniform distribution
    dist = UniformCube(n_dimensions)
    
    # Monte Carlo (MC) experiment
    mc_sampler = UnitCubeSampler()
    mc_results = run_sampling_experiment(
        distribution=dist,
        target_function=model_prediction,
        sampler=mc_sampler,
        n_samples=n_samples,
        n_dimensions=n_dimensions
    )
    
    # Quasi-Monte Carlo (QMC) experiment
    qmc_sampler = SobolSampler(scramble=True)
    qmc_results = run_sampling_experiment(
        distribution=dist,
        target_function=model_prediction,
        sampler=qmc_sampler,
        n_samples=n_samples,
        n_dimensions=n_dimensions
    )
    
    # Print results
    print(f"\nResults for {n_dimensions} dimensions:")
    print(f"Number of samples: {n_samples}")
    print(f"MC  estimate: {mc_results['expectation']:.6f}")
    print(f"QMC estimate: {qmc_results['expectation']:.6f}")
    print(f"MC  variance: {mc_results['variance']:.6f}")
    print(f"QMC variance: {qmc_results['variance']:.6f}")
    
    return mc_results, qmc_results

# Parameters
n_dimensions = X_train.shape[1]  # Number of features
n_samples = 2**9  # Use power of 2 for Sobol sequences

# Run the experiment and visualize convergence
mc_results, qmc_results = run_comparison_medical(n_dimensions, n_samples)

plt.figure(figsize=(10, 6))

# MC Convergence Plot
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

# QMC Convergence Plot
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

# Final Plot Adjustments
plt.axhline(y=np.mean(y_pred), color='r', linestyle='--', label='Test Set Mean Prediction')
plt.xlabel('Number of Samples')
plt.ylabel('Predicted Medical Cost Estimate')
plt.title('MC vs QMC Convergence for Medical Cost Prediction')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
plt.savefig('examples/medical_cost_qmc_comparison.png')
