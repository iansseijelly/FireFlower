import os
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from tacit_learn.dataloader import BasicBlockDataset, collate_fn, create_train_val_dataloaders
from tacit_learn.model import FireFlowerPredictor, FireFlowerConfig
from math import sqrt
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import random
import glob
from scipy import stats  # Add this import
import pickle

import constants

# Create result directories
os.makedirs(constants.RESULTS_DIR, exist_ok=True)
os.makedirs(os.path.join(constants.RESULTS_DIR, 'data'), exist_ok=True)
os.makedirs(os.path.join(constants.RESULTS_DIR, 'human'), exist_ok=True)

# =====================
#   Dataset & DataLoader
# =====================
training_files = glob.glob(os.path.join(constants.DATA_FOLDER, "*.out"))

# Create train and validation datasets
train_loader, val_loader = create_train_val_dataloaders(
    vocab_path=constants.VOCAB_FILE,
    file_paths=training_files,
    batch_size=constants.BATCH_SIZE,
    val_ratio=0.1,  # Same ratio as used in training
    max_block_len=constants.MAX_BLOCK_LEN,
    shuffle_train=False,  # No need to shuffle for evaluation
    seed=42  # Same seed as training for consistent split
)

# Use validation set for evaluation
test_loader = val_loader

# Get access to the original dataset to determine vocabulary sizes
test_dataset = test_loader.dataset.dataset  # Access the original dataset through the random_split subset
while hasattr(test_dataset, 'dataset'):
    test_dataset = test_dataset.dataset

# =====================
#   Load Model
# =====================
# Initialize model with the same configuration as during training
NUM_INST = test_dataset.get_n_inst()
NUM_REGS = test_dataset.get_n_reg()
MAX_BLOCK_LEN = constants.MAX_BLOCK_LEN
config = FireFlowerConfig(n_inst=NUM_INST, 
                          n_reg=NUM_REGS,
                          d_inst=int(sqrt(NUM_INST)),
                          d_reg=int(sqrt(NUM_REGS)),
                          d_imm=int(sqrt(NUM_INST)),
                          d_bb=int(sqrt(NUM_INST)),
                          d_model=constants.D_MODEL,
                          n_head=constants.N_HEAD,
                          n_layers=constants.N_LAYERS,
                          n_pos=MAX_BLOCK_LEN,
                          d_pos=int(sqrt(NUM_REGS)))

model = FireFlowerPredictor(config)
if torch.__version__ >= "2.0.0":
    model = torch.compile(model)

# Try to load the best model from training, fall back to specified checkpoint path
if os.path.exists('checkpoints/best_model.pth'):
    checkpoint = torch.load('checkpoints/best_model.pth', map_location=constants.DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded best model from epoch {checkpoint['epoch']} with validation loss {checkpoint.get('val_loss', 'N/A')}")
else:
    model.load_state_dict(torch.load(constants.CHECKPOINT_PATH, map_location=constants.DEVICE))
    print(f"Loaded model from {constants.CHECKPOINT_PATH}")

model.to(constants.DEVICE)
model.eval()  # Set model to evaluation mode

# =====================
#   Evaluation Metrics
# =====================
def calculate_metrics(predictions, targets, bb_predictions, bb_times):
    """Calculate various metrics for model evaluation."""
    # Flatten predictions and targets for instruction-level metrics
    pred_flat = predictions.reshape(-1).numpy()
    target_flat = targets.reshape(-1).numpy()
    
    # Filter out padded values (assuming 0 is a padding indicator)
    mask = target_flat != 0
    pred_filtered = pred_flat[mask]
    target_filtered = target_flat[mask]
    
    # Basic block level metrics
    bb_pred = bb_predictions.reshape(-1).numpy()
    bb_target = bb_times.reshape(-1).numpy()
    
    metrics = {
        # Instruction-level metrics
        "instr_mse": mean_squared_error(target_filtered, pred_filtered),
        "instr_rmse": np.sqrt(mean_squared_error(target_filtered, pred_filtered)),
        "instr_mae": mean_absolute_error(target_filtered, pred_filtered),
        "instr_r2": r2_score(target_filtered, pred_filtered),
        "instr_corr": np.corrcoef(target_filtered, pred_filtered)[0, 1],
        "instr_mean_error": np.mean(pred_filtered - target_filtered),
        "instr_mean_abs_error": np.mean(np.abs(pred_filtered - target_filtered)),
        "instr_mean_rel_error": np.mean(np.abs(pred_filtered - target_filtered) / (target_filtered + 1e-8)),
        
        # Basic block level metrics
        "bb_mse": mean_squared_error(bb_target, bb_pred),
        "bb_rmse": np.sqrt(mean_squared_error(bb_target, bb_pred)),
        "bb_mae": mean_absolute_error(bb_target, bb_pred),
        "bb_r2": r2_score(bb_target, bb_pred),
        "bb_corr": np.corrcoef(bb_target, bb_pred)[0, 1],
        "bb_mean_error": np.mean(bb_pred - bb_target),
        "bb_mean_abs_error": np.mean(np.abs(bb_pred - bb_target)),
        "bb_mean_rel_error": np.mean(np.abs(bb_pred - bb_target) / (bb_target + 1e-8)),
    }
    
    return metrics, pred_filtered, target_filtered, bb_pred, bb_target

# =====================
#   OLS Regression
# =====================
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
    
    plt.scatter(x_jitter, y_jitter, alpha=0.3)
    
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
    plt.savefig(os.path.join(constants.RESULTS_DIR, f'{level}_predictions_vs_targets.png'))
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
    plt.savefig(os.path.join(constants.RESULTS_DIR, f'{level}_error_distribution.png'))
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
    plt.savefig(os.path.join(constants.RESULTS_DIR, f'{level}_relative_error.png'))
    plt.close()

# =====================
#   Run Evaluation
# =====================

def format_results(results):
    """Format the prediction results in a human-readable way."""
    output = []
    
    for block in results:
        output.append(f"\n===== Basic Block {block['block_idx']} =====")
        output.append(f"Total Predicted Time: {block['total_predicted']:.4f}")
        output.append(f"Total Actual Time: {block['total_actual']:.4f}")
        output.append(f"Difference: {block['total_predicted'] - block['total_actual']:.4f}")
        output.append(f"Relative Error: {abs(block['total_predicted'] - block['total_actual']) / (block['total_actual'] + 1e-8):.4f}\n")
        
        # Table header
        output.append(f"{'Position':<8} {'Opcode':<15} {'Predicted':<12} {'Actual':<12} {'Diff':<10} {'Rel Error':<10} {'Instruction'}")
        output.append("-" * 100)
        
        # Table rows
        for instr in block["instructions"]:
            actual = instr['actual_latency']
            predicted = instr['predicted_latency']
            
            if actual is not None:
                diff = predicted - actual
                rel_error = abs(diff) / (actual + 1e-8)
                actual_str = f"{actual:.4f}"
                diff_str = f"{diff:.4f}"
                rel_error_str = f"{rel_error:.4f}"
            else:
                actual_str = "N/A"
                diff_str = "N/A"
                rel_error_str = "N/A"
                
            output.append(
                f"{instr['position']:<8} {instr['opcode'][:15]:<15} {predicted:.4f}       "
                f"{actual_str:<12} {diff_str:<10} {rel_error_str:<10} {instr['instruction']}"
            )
            
        # Add separator between blocks
        output.append("\n" + "=" * 100)
    
    return "\n".join(output)


def run_evaluation():
    """Run evaluation with proper handling of variable sequence lengths."""
    print(f"Evaluating model on validation set ({len(test_loader.dataset)} samples)")
    
    all_predictions = []
    all_block_data = []
    
    total_batches = len(test_loader)
    # Determine how many batches to process
    batches_to_process = constants.SAMPLE_BATCHES if constants.SAMPLE_BATCHES and constants.SAMPLE_BATCHES < total_batches else total_batches
    print(f"Evaluating on {batches_to_process} batches out of {total_batches} total batches")

    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            # Stop after processing the specified number of batches
            if constants.SAMPLE_BATCHES and batch_idx >= constants.SAMPLE_BATCHES:
                break
                
            # Move batch to constants.DEVICE
            instructions = {k: v.to(constants.DEVICE) for k, v in batch["instructions"].items()}
            positions = batch["positions"].to(constants.DEVICE)
            bbtime = batch["bbtime"].float().to(constants.DEVICE)  # Convert to float
            target = batch["target"].float().to(constants.DEVICE)  # Convert to float
            padding_mask = batch["padding_mask"].to(constants.DEVICE)
            
            # Forward pass
            preds = model(instructions, positions, bbtime)

            # Apply mask to predictions and targets
            masked_preds = preds * padding_mask
            masked_target = target * padding_mask
            
            # Calculate basic block predictions (sum of instruction predictions)
            bb_preds = masked_preds.sum(dim=1)
            
            # Store data for each block in batch
            for i in range(masked_preds.size(0)):  # Iterate through each sample in batch
                block_data = {
                    "block_idx": batch_idx * constants.BATCH_SIZE + i,
                    "total_predicted": bb_preds[i].item(),
                    "total_actual": bbtime[i].item(),
                    "instructions": []
                }
                
                # Get instruction details
                for j in range(masked_preds.size(1)):  # Iterate through each instruction
                    if j < masked_target.size(1) and masked_target[i, j, 0] != 0:  # Skip padding
                        # Try to get instruction text if available
                        instr_text = batch.get("instruction_text", [["Unknown"]])[i][j] if j < len(batch.get("instruction_text", [[]])[i]) else "Unknown"
                        # Try to get opcode if available
                        opcode = "Unknown"
                        if "inst_id" in instructions:
                            inst_id = instructions["inst_id"][i, j].item()
                            if hasattr(test_dataset, "id2inst"):
                                opcode = test_dataset.id2inst.get(inst_id, "Unknown")
                        
                        block_data["instructions"].append({
                            "position": j,
                            "opcode": opcode,
                            "instruction": instr_text,
                            "predicted_latency": masked_preds[i, j, 0].item(),
                            "actual_latency": masked_target[i, j, 0].item()
                        })
                
                all_block_data.append(block_data)
            
            # Collect data for metrics calculation
            batch_predictions = {
                "instruction_predictions": masked_preds.cpu().numpy(),
                "instruction_targets": masked_target.cpu().numpy(),
                "bb_predictions": bb_preds.cpu().numpy(),
                "bb_targets": bbtime.cpu().numpy(),
                "padding_mask": padding_mask.cpu().numpy()
            }
            all_predictions.append(batch_predictions)
            
            if batch_idx % 10 == 0:
                print(f"Processed {batch_idx+1}/{batches_to_process} batches")
    
    # Save the raw prediction data in compressed format
    data_path = os.path.join(constants.RESULTS_DIR, 'data', 'prediction_data.pkl')
    with open(data_path, 'wb') as f:
        pickle.dump(all_predictions, f)
    
    # Also save the structured block data for easier access
    block_data_path = os.path.join(constants.RESULTS_DIR, 'data', 'block_data.pkl')
    with open(block_data_path, 'wb') as f:
        pickle.dump(all_block_data, f)
        
    # Save human-readable formatted results
    human_readable_path = os.path.join(constants.RESULTS_DIR, 'human', 'predictions.txt')
    with open(human_readable_path, 'w') as f:
        f.write(format_results(all_block_data))
    
    print(f"\nEvaluation complete.")
    print(f"Raw prediction data saved to: {data_path}")
    print(f"Block data saved to: {block_data_path}")
    print(f"Human-readable results saved to: {human_readable_path}")
    
    return all_block_data, all_predictions

if __name__ == "__main__":
    run_evaluation() 