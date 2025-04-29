import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy import stats
import pickle
from scipy.ndimage import gaussian_filter

import constants

# Create directories for visualization results
os.makedirs(constants.RESULTS_DIR, exist_ok=True)
os.makedirs(os.path.join(constants.RESULTS_DIR, 'metrics'), exist_ok=True)

def load_prediction_data():
    """Load prediction data saved by evaluate.py"""
    block_data_path = os.path.join(constants.RESULTS_DIR, 'data', 'block_data.pkl')
    predictions_path = os.path.join(constants.RESULTS_DIR, 'data', 'prediction_data.pkl')
    
    if not os.path.exists(block_data_path) or not os.path.exists(predictions_path):
        raise FileNotFoundError(
            "Prediction data not found. Please run evaluate.py first to generate predictions."
        )
    
    with open(block_data_path, 'rb') as f:
        block_data = pickle.load(f)
    
    with open(predictions_path, 'rb') as f:
        prediction_data = pickle.load(f)
    
    return block_data, prediction_data

def prepare_data_for_metrics(block_data, prediction_data):
    """Extract and prepare data for metrics calculation"""
    # Extract instruction-level data from block_data
    pred_instr = np.array([instr["predicted_latency"] for block in block_data for instr in block["instructions"]])
    target_instr = np.array([instr["actual_latency"] for block in block_data for instr in block["instructions"]])
    
    # Extract basic block level data
    bb_preds = np.array([block["total_predicted"] for block in block_data])
    bb_targets = np.array([block["total_actual"] for block in block_data])
    
    return pred_instr, target_instr, bb_preds, bb_targets

# =====================
#   Metrics Functions
# =====================
def calculate_metrics(pred_instr, target_instr, bb_preds, bb_targets):
    """Calculate various metrics for model evaluation."""
    metrics = {
        # Instruction-level metrics
        "instr_mse": mean_squared_error(target_instr, pred_instr),
        "instr_rmse": np.sqrt(mean_squared_error(target_instr, pred_instr)),
        "instr_mae": mean_absolute_error(target_instr, pred_instr),
        "instr_r2": r2_score(target_instr, pred_instr),
        "instr_corr": np.corrcoef(target_instr, pred_instr)[0, 1],
        "instr_mean_error": np.mean(pred_instr - target_instr),
        "instr_mean_abs_error": np.mean(np.abs(pred_instr - target_instr)),
        "instr_mean_rel_error": np.mean(np.abs(pred_instr - target_instr) / (target_instr + 1e-8)),
        
        # Basic block level metrics
        "bb_mse": mean_squared_error(bb_targets, bb_preds),
        "bb_rmse": np.sqrt(mean_squared_error(bb_targets, bb_preds)),
        "bb_mae": mean_absolute_error(bb_targets, bb_preds),
        "bb_r2": r2_score(bb_targets, bb_preds),
        "bb_corr": np.corrcoef(bb_targets, bb_preds)[0, 1],
        "bb_mean_error": np.mean(bb_preds - bb_targets),
        "bb_mean_abs_error": np.mean(np.abs(bb_preds - bb_targets)),
        "bb_mean_rel_error": np.mean(np.abs(bb_preds - bb_targets) / (bb_targets + 1e-8)),
    }
    
    return metrics

def ols_regression(predictions, targets):
    """Perform OLS regression on predictions and targets."""
    # Perform linear regression
    slope, intercept, r_value, p_value, std_err = stats.linregress(predictions, targets)
    
    # Calculate predicted values using the regression line
    predicted = slope * predictions + intercept
    
    # Calculate additional metrics
    mse = mean_squared_error(targets, predicted)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(targets, predicted)
    nrmse = rmse / (targets.max() - targets.min())
    
    # Return all statistics and predicted values
    return {
        'slope': slope,
        'intercept': intercept,
        'r_squared': r_value**2,
        'r_value': r_value,
        'p_value': p_value,
        'std_err': std_err,
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'nrmse': nrmse,
    }

# =====================
#   Visualization Functions
# =====================
def plot_predictions_vs_targets(predictions, targets, level="instruction", jitter=False):
    """Create scatter plot of predictions vs targets."""
    plt.figure(figsize=(10, 8))
    
    # Add small jitter to spread out discrete values
    if jitter:
        jitter_amount = 0.03  # Adjust this value based on your data scale
        x_jitter = targets + np.random.normal(0, jitter_amount, size=targets.shape)
        y_jitter = predictions + np.random.normal(0, jitter_amount, size=predictions.shape)
    else:
        x_jitter = targets
        y_jitter = predictions
    
    # Ensure positive values for log scale (add small epsilon)
    epsilon = 1e-10
    x_jitter = np.maximum(x_jitter, epsilon)
    y_jitter = np.maximum(y_jitter, epsilon)
    
    plt.scatter(x_jitter, y_jitter, alpha=0.1)
    
    # Set log scale for both axes
    plt.xscale('log')
    plt.yscale('log')
    
    # Plot the perfect prediction line
    min_val = min(np.min(predictions), np.min(targets))
    max_val = max(np.max(predictions), np.max(targets))
    min_val = max(min_val, epsilon)  # Ensure positive for log scale
    plt.plot([min_val, max_val], [min_val, max_val], 'r--')
    
    plt.xlabel(f'Actual {level} latency')
    plt.ylabel(f'Predicted {level} latency')
    plt.title(f'{level.capitalize()} Level: Predicted vs Actual Latency (Log Scale)')
    plt.grid(True, alpha=0.3)
    
    # Add correlation coefficient
    corr = np.corrcoef(targets, predictions)[0, 1]
    plt.annotate(f'Correlation: {corr:.4f}', 
                 xy=(0.05, 0.95), 
                 xycoords='axes fraction',
                 bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(os.path.join(constants.RESULTS_DIR, 'metrics', f'{level}_predictions_vs_targets.png'))
    plt.close()

def plot_error_distribution(predictions, targets, level="instruction"):
    """Plot the distribution of prediction errors."""
    errors = predictions - targets
    
    plt.figure(figsize=(10, 6))
    plt.hist(errors, bins=50, alpha=0.7)
    plt.axvline(x=0, color='r', linestyle='--')
    plt.xlabel('Prediction Error')
    plt.ylabel('Frequency')
    plt.title(f'{level.capitalize()} Level: Error Distribution')
    plt.grid(True, alpha=0.3)
    
    # Add mean and std annotations
    mean_error = np.mean(errors)
    std_error = np.std(errors)
    plt.annotate(f'Mean Error: {mean_error:.4f}\nStd Dev: {std_error:.4f}', 
                 xy=(0.05, 0.95), 
                 xycoords='axes fraction',
                 bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(os.path.join(constants.RESULTS_DIR, 'metrics', f'{level}_error_distribution.png'))
    plt.close()

def plot_relative_error(predictions, targets, level="instruction"):
    """Plot the relative error distribution."""
    rel_errors = np.abs(predictions - targets) / (targets + 1e-8)
    # Clip extreme values for better visualization
    rel_errors = np.clip(rel_errors, 0, 5)
    
    plt.figure(figsize=(10, 6))
    plt.hist(rel_errors, bins=50, alpha=0.7)
    plt.xlabel('Relative Error (|pred-target|/target)')
    plt.ylabel('Frequency')
    plt.title(f'{level.capitalize()} Level: Relative Error Distribution')
    plt.grid(True, alpha=0.3)
    
    # Add mean relative error annotation
    mean_rel_error = np.mean(rel_errors)
    plt.annotate(f'Mean Relative Error: {mean_rel_error:.4f}', 
                 xy=(0.05, 0.95), 
                 xycoords='axes fraction',
                 bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(os.path.join(constants.RESULTS_DIR, 'metrics', f'{level}_relative_error.png'))
    plt.close()

def generate_visualizations():
    """Generate all visualizations and metrics from prediction data."""
    print("Loading prediction data...")
    block_data, prediction_data = load_prediction_data()
    
    print("Preparing data for metrics calculation...")
    pred_instr, target_instr, bb_preds, bb_targets = prepare_data_for_metrics(block_data, prediction_data)
    
    print("Calculating metrics...")
    metrics = calculate_metrics(pred_instr, target_instr, bb_preds, bb_targets)
    
    # Print metrics
    print("\n===== Evaluation Results =====")
    print("\nInstruction-level Metrics:")
    for key in sorted([k for k in metrics.keys() if k.startswith('instr')]):
        print(f"{key}: {metrics[key]:.6f}")
    
    print("\nBasic Block-level Metrics:")
    for key in sorted([k for k in metrics.keys() if k.startswith('bb')]):
        print(f"{key}: {metrics[key]:.6f}")
    
    # Perform OLS regression
    instr_ols_results = ols_regression(pred_instr, target_instr)
    bb_ols_results = ols_regression(bb_preds, bb_targets)
    
    print("\nInstruction-level OLS Regression Results:")
    for key, value in instr_ols_results.items():
        print(f"{key}: {value:.6f}")
    
    print("\nBasic Block-level OLS Regression Results:")
    for key, value in bb_ols_results.items():
        print(f"{key}: {value:.6f}")
    
    # Create visualizations
    print("\nGenerating visualizations...")
    plot_predictions_vs_targets(pred_instr, target_instr, level="instruction", jitter=True)
    plot_error_distribution(pred_instr, target_instr, level="instruction")
    plot_relative_error(pred_instr, target_instr, level="instruction")
    
    plot_predictions_vs_targets(bb_preds, bb_targets, level="basic_block")
    # plot_heatmap(bb_preds, bb_targets, level="basic_block")
    # plot_error_distribution(bb_preds, bb_targets, level="basic_block")
    # plot_relative_error(bb_preds, bb_targets, level="basic_block")
    
    # Save metrics to CSV
    metrics_df = pd.DataFrame([metrics])
    metrics_df.to_csv(os.path.join(constants.RESULTS_DIR, 'metrics', 'evaluation_metrics.csv'), index=False)
    
    # Save OLS results to CSV
    instr_ols_df = pd.DataFrame([instr_ols_results])
    bb_ols_df = pd.DataFrame([bb_ols_results])
    instr_ols_df.to_csv(os.path.join(constants.RESULTS_DIR, 'metrics', 'instruction_ols_results.csv'), index=False)
    bb_ols_df.to_csv(os.path.join(constants.RESULTS_DIR, 'metrics', 'basic_block_ols_results.csv'), index=False)
    
    print(f"\nVisualization complete. Results saved to {os.path.join(constants.RESULTS_DIR, 'metrics')}")
    
    return metrics, instr_ols_results, bb_ols_results

if __name__ == "__main__":
    generate_visualizations() 