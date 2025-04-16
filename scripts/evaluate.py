import os
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from tacit_learn.dataloader import BasicBlockDataset, collate_fn
from tacit_learn.model import FireFlowerPredictor, FireFlowerConfig
from math import sqrt
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# =====================
#   Configuration
# =====================
CHECKPOINT_PATH = "checkpoints/model_final.pth"  # Path to your trained model
DATA_FILE = "data/rocket-hello.canonicalized.out"  # Test/validation data file
VOCAB_FILE = "vocab/opcodes.txt"
BATCH_SIZE = 16  # Can be larger for evaluation than training
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RESULTS_DIR = "./evaluation_results"
os.makedirs(RESULTS_DIR, exist_ok=True)

# =====================
#   Dataset & DataLoader
# =====================
test_dataset = BasicBlockDataset(vocab_path=VOCAB_FILE, file_path=DATA_FILE)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, collate_fn=collate_fn, shuffle=False)

# =====================
#   Load Model
# =====================
# Initialize model with the same configuration as during training
NUM_INST = test_dataset.get_n_inst()
NUM_REGS = test_dataset.get_n_reg()
MAX_BLOCK_LEN = 64
config = FireFlowerConfig(n_inst=NUM_INST, 
                          n_reg=NUM_REGS,
                          d_inst=int(sqrt(NUM_INST)),
                          d_reg=int(sqrt(NUM_REGS)),
                          d_imm=int(sqrt(NUM_INST)),
                          d_bb=int(sqrt(NUM_INST)),
                          d_model=128,
                          n_head=4,
                          n_layers=2,
                          n_pos=MAX_BLOCK_LEN,
                          d_pos=int(sqrt(NUM_REGS)))

model = FireFlowerPredictor(config)
model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE))
model.to(DEVICE)
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
#   Visualization Functions
# =====================
def plot_predictions_vs_targets(predictions, targets, level="instruction"):
    """Create scatter plot of predictions vs targets."""
    plt.figure(figsize=(10, 8))
    plt.scatter(targets, predictions, alpha=0.5)
    
    # Plot the perfect prediction line
    min_val = min(np.min(predictions), np.min(targets))
    max_val = max(np.max(predictions), np.max(targets))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--')
    
    plt.xlabel(f'Actual {level} latency')
    plt.ylabel(f'Predicted {level} latency')
    plt.title(f'{level.capitalize()} Level: Predicted vs Actual Latency')
    plt.grid(True, alpha=0.3)
    
    # Add correlation coefficient
    corr = np.corrcoef(targets, predictions)[0, 1]
    plt.annotate(f'Correlation: {corr:.4f}', 
                 xy=(0.05, 0.95), 
                 xycoords='axes fraction',
                 bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, f'{level}_predictions_vs_targets.png'))
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
    plt.savefig(os.path.join(RESULTS_DIR, f'{level}_error_distribution.png'))
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
    plt.savefig(os.path.join(RESULTS_DIR, f'{level}_relative_error.png'))
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
    all_preds = []
    all_targets_flat = []
    all_bb_predictions = []
    all_bb_times = []
    
    print("Running evaluation...")
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            # Move batch to device
            instructions = {k: v.to(DEVICE) for k, v in batch["instructions"].items()}
            positions = batch["positions"].to(DEVICE)
            bbtime = batch["bbtime"].float().to(DEVICE)  # Convert to float
            target = batch["target"].float().to(DEVICE)  # Convert to float
            
            # Forward pass
            preds = model(instructions, positions, bbtime)
            
            # Calculate basic block predictions (sum of instruction predictions)
            bb_preds = preds.sum(dim=1)
            
            # Store prediction results for formatting
            for i in range(preds.size(0)):  # Iterate through each sample in batch
                block_data = {
                    "block_idx": batch_idx * BATCH_SIZE + i,
                    "total_predicted": bb_preds[i].item(),
                    "total_actual": bbtime[i].item(),
                    "instructions": []
                }
                
                # Get instruction details
                for j in range(preds.size(1)):  # Iterate through each instruction
                    if j < target.size(1) and target[i, j, 0] != 0:  # Skip padding
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
                            "predicted_latency": preds[i, j, 0].item(),
                            "actual_latency": target[i, j, 0].item()
                        })
                
                all_preds.append(block_data)
            
            # Immediately flatten and filter out padding before storing
            # This avoids the dimension mismatch when concatenating later
            batch_mask = target.reshape(-1) != 0  # Filter out padding (assumes 0 is padding)
            
            # Collect flattened results for metrics calculation
            all_targets_flat.append(target.reshape(-1)[batch_mask].cpu())
            all_bb_predictions.append(bb_preds.cpu())
            all_bb_times.append(bbtime.cpu())
            
            if batch_idx % 10 == 0:
                print(f"Processed {batch_idx+1}/{len(test_loader)} batches")

    # save all predictions and targets in pretty text format
    with open(os.path.join(RESULTS_DIR, 'predictions.txt'), 'w') as f:
        f.write(format_results(all_preds))
    
    # Concatenate all predictions and targets for metrics
    all_targets_flat = torch.cat(all_targets_flat, dim=0)
    all_bb_predictions = torch.cat(all_bb_predictions, dim=0)
    all_bb_times = torch.cat(all_bb_times, dim=0)
    
    # Convert to numpy for metric calculation
    pred_instr = torch.tensor([instr["predicted_latency"] for block in all_preds for instr in block["instructions"]]).numpy()
    target_instr = torch.tensor([instr["actual_latency"] for block in all_preds for instr in block["instructions"]]).numpy()
    bb_preds = all_bb_predictions.reshape(-1).numpy()
    bb_targets = all_bb_times.reshape(-1).numpy()
    
    # Calculate metrics directly
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
    
    # Print metrics
    print("\n===== Evaluation Results =====")
    print("\nInstruction-level Metrics:")
    for key in sorted([k for k in metrics.keys() if k.startswith('instr')]):
        print(f"{key}: {metrics[key]:.6f}")
    
    print("\nBasic Block-level Metrics:")
    for key in sorted([k for k in metrics.keys() if k.startswith('bb')]):
        print(f"{key}: {metrics[key]:.6f}")
    
    # Create visualizations
    print("\nGenerating visualizations...")
    plot_predictions_vs_targets(pred_instr, target_instr, level="instruction")
    plot_error_distribution(pred_instr, target_instr, level="instruction")
    plot_relative_error(pred_instr, target_instr, level="instruction")
    
    plot_predictions_vs_targets(bb_preds, bb_targets, level="basic_block")
    plot_error_distribution(bb_preds, bb_targets, level="basic_block")
    plot_relative_error(bb_preds, bb_targets, level="basic_block")
    
    # Save metrics to CSV
    metrics_df = pd.DataFrame([metrics])
    metrics_df.to_csv(os.path.join(RESULTS_DIR, 'evaluation_metrics.csv'), index=False)
    
    print(f"\nEvaluation complete. Results saved to {RESULTS_DIR}")
    
    return metrics

if __name__ == "__main__":
    run_evaluation() 