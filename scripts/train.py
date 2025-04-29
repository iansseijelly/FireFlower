import os
import datetime
from math import sqrt
import glob
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# Import your dataset and model.
# Adjust the import paths if necessary.
from tacit_learn.dataloader import BasicBlockDataset, collate_fn, create_train_val_dataloaders
from tacit_learn.model import FireFlowerPredictor, FireFlowerConfig

import constants

# =====================
#   Dataset & DataLoader
# =====================
training_files = glob.glob(os.path.join(constants.DATA_FOLDER, "*.out"))

# Replace the existing dataset and dataloader setup with our new function
train_loader, val_loader = create_train_val_dataloaders(
    vocab_path=constants.VOCAB_FILE,
    file_paths=training_files,
    batch_size=constants.BATCH_SIZE,
    val_ratio=0.1,  # Use 10% of data for validation
    max_block_len=constants.MAX_BLOCK_LEN,
    shuffle_train=True,
    seed=42  # For reproducibility
)

# Get a sample from the first batch to determine vocabulary sizes
sample_batch = next(iter(train_loader))
if hasattr(train_loader.dataset.dataset, 'get_n_inst'):
    # Access the original dataset through the random_split subset
    NUM_INST = train_loader.dataset.dataset.get_n_inst()
    NUM_REGS = train_loader.dataset.dataset.get_n_reg()
else:
    # Fallback if structure is different
    first_dataset = train_loader.dataset
    while hasattr(first_dataset, 'dataset'):
        first_dataset = first_dataset.dataset
    NUM_INST = first_dataset.get_n_inst()
    NUM_REGS = first_dataset.get_n_reg()

# =====================
#   Instantiate the Model
# =====================
config = FireFlowerConfig(n_inst=NUM_INST, 
                          n_reg=NUM_REGS,
                          d_inst=int(sqrt(NUM_INST)),
                          d_reg=int(sqrt(NUM_REGS)),
                          d_imm=int(sqrt(NUM_INST)),
                          d_bb=int(sqrt(NUM_INST)),
                          d_model=constants.D_MODEL,
                          n_head=constants.N_HEAD,
                          n_layers=constants.N_LAYERS,
                          n_pos=constants.MAX_BLOCK_LEN,
                          d_pos=int(sqrt(NUM_REGS)))

torch.set_float32_matmul_precision('high')

model = FireFlowerPredictor(config)

if torch.__version__ >= "2.0.0":
    model = torch.compile(model)
else:
    print("Warning: torch version is less than 2.0.0, model will not be compiled.")

model.to(constants.DEVICE)


# =====================
#   Loss & Optimizer
# =====================
mse_loss = nn.MSELoss()
optimizer = optim.AdamW(model.parameters(), lr=constants.LEARNING_RATE)
max_grad_norm = 1.0

# =====================
#   TensorBoard Setup
# =====================
writer = SummaryWriter(constants.LOG_DIR)

# Function to log model parameter statistics
def log_model_stats(model, step):
    for name, param in model.named_parameters():
        if param.requires_grad:
            writer.add_histogram(f"parameters/{name}", param.data, step)
            if param.grad is not None:
                writer.add_histogram(f"gradients/{name}", param.grad.data, step)
            writer.add_scalar(f"parameter_norm/{name}", param.data.norm().item(), step)
            if param.grad is not None:
                writer.add_scalar(f"gradient_norm/{name}", param.grad.data.norm().item(), step)

# =====================
#   Evaluation Function
# =====================
def evaluate(model, val_loader, device):
    model.eval()
    val_loss = 0.0
    val_reg_loss = 0.0
    val_bb_loss = 0.0
    val_fp_loss = 0.0
    val_fn_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for batch in val_loader:
            # Move batch to device.
            instructions = {k: v.to(device) for k, v in batch["instructions"].items()}
            positions = batch["positions"].to(device)
            bbtime = batch["bbtime"].to(device)  # shape: [B, 1]
            target = batch["target"].to(device)   # shape: [B, L, 1]
            mask = batch["padding_mask"].to(device)

            # Forward pass.
            preds = model(instructions, positions, bbtime)  # shape: [B, L, 1]

            # Apply mask to predictions and targets
            masked_preds = preds * mask
            masked_target = target * mask
            
            # Count valid (non-padding) instructions for proper averaging
            num_valid = mask.sum()
            
            # Compute per-instruction regression loss (only on valid instructions)
            if num_valid > 0:
                reg_loss = torch.sum(((masked_preds - masked_target) ** 2)) / num_valid
            else:
                reg_loss = torch.tensor(0.0, device=device)

            # False positives and negatives
            false_pos = torch.sum((torch.round(masked_preds) == 1) & (masked_target != 1)) / num_valid if num_valid > 0 else torch.tensor(0.0, device=device)
            false_neg = torch.sum((torch.round(masked_preds) != 1) & (masked_target == 1)) / num_valid if num_valid > 0 else torch.tensor(0.0, device=device)

            # Compute block-level sum loss.
            pred_bbtime = masked_preds.sum(dim=1)  # shape: [B, 1]
            bb_loss = mse_loss(pred_bbtime, bbtime)

            # Combined loss
            loss = constants.REG_LOSS_WEIGHT * reg_loss + constants.BB_LOSS_WEIGHT * bb_loss + \
                   constants.FP_LOSS_WEIGHT * false_pos + constants.FN_LOSS_WEIGHT * false_neg
            
            val_loss += loss.item()
            val_reg_loss += reg_loss.item()
            val_bb_loss += bb_loss.item()
            val_fp_loss += false_pos.item()
            val_fn_loss += false_neg.item()
            num_batches += 1
    
    # Calculate average losses
    val_loss /= num_batches
    val_reg_loss /= num_batches
    val_bb_loss /= num_batches
    val_fp_loss /= num_batches
    val_fn_loss /= num_batches
    
    return {
        "loss": val_loss,
        "reg_loss": val_reg_loss,
        "bb_loss": val_bb_loss,
        "fp_loss": val_fp_loss,
        "fn_loss": val_fn_loss
    }

# =====================
#   Training Loop
# =====================
global_step = 0
best_val_loss = float('inf')
patience = constants.PATIENCE
patience_counter = 0

for epoch in range(constants.NUM_EPOCHS):
    # Training phase
    model.train()
    running_loss = 0.0
    # Log learning rate at start of epoch
    writer.add_scalar("Learning_Rate", optimizer.param_groups[0]['lr'], epoch)
    
    for batch_idx, batch in enumerate(train_loader):
        # Move batch to device.
        instructions = {k: v.to(constants.DEVICE) for k, v in batch["instructions"].items()}
        positions = batch["positions"].to(constants.DEVICE)
        bbtime = batch["bbtime"].to(constants.DEVICE)  # shape: [B, 1]
        target = batch["target"].to(constants.DEVICE)   # shape: [B, L, 1]
        mask = batch["padding_mask"].to(constants.DEVICE)

        optimizer.zero_grad()

        # Forward pass.
        preds = model(instructions, positions, bbtime)  # shape: [B, L, 1]

        # Apply mask to predictions and targets
        masked_preds = preds * mask
        masked_target = target * mask
        
        # Count valid (non-padding) instructions for proper averaging
        num_valid = mask.sum()
        
        # Compute per-instruction regression loss (only on valid instructions)
        if num_valid > 0:
            reg_loss = torch.sum(((masked_preds - masked_target) ** 2)) / num_valid
        else:
            reg_loss = torch.tensor(0.0, device=constants.DEVICE)

        # penalize for incorrectly predicting a 1 as not a 1, and vice versa
        false_pos = torch.sum((torch.round(masked_preds) == 1) & (masked_target != 1)) / num_valid if num_valid > 0 else torch.tensor(0.0, device=constants.DEVICE)
        false_neg = torch.sum((torch.round(masked_preds) != 1) & (masked_target == 1)) / num_valid if num_valid > 0 else torch.tensor(0.0, device=constants.DEVICE)

        # Compute block-level sum loss.
        # Sum predictions over instructions (dim=1) and compare with bbtime.
        pred_bbtime = masked_preds.sum(dim=1)  # shape: [B, 1]
        bb_loss = mse_loss(pred_bbtime, bbtime)

        # After computing losses
        if torch.isnan(reg_loss) or torch.isnan(bb_loss):
            print(f"NaN detected! reg_loss: {reg_loss.item()}, bb_loss: {bb_loss.item()}")
            print(f"preds min/max: {preds.min().item()}/{preds.max().item()}")
            print(f"target min/max: {target.min().item()}/{target.max().item()}")
            print(f"bbtime min/max: {bbtime.min().item()}/{bbtime.max().item()}")
            # Optional: skip this batch
            breakpoint()

        loss = constants.REG_LOSS_WEIGHT * reg_loss + constants.BB_LOSS_WEIGHT * bb_loss + \
                constants.FP_LOSS_WEIGHT * false_pos + constants.FN_LOSS_WEIGHT * false_neg

        # Backpropagation.
        loss.backward()
        
        # Calculate gradient norm before clipping (for monitoring)
        total_norm = 0
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5
        writer.add_scalar("Gradient/TotalNormBeforeClip", total_norm, global_step)
        
        # Clip gradients to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        
        # Log gradient norm after clipping
        total_norm_after = 0
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm_after += param_norm.item() ** 2
        total_norm_after = total_norm_after ** 0.5
        writer.add_scalar("Gradient/TotalNormAfterClip", total_norm_after, global_step)
        
        optimizer.step()

        running_loss += loss.item()
        global_step += 1

        # Log every few batches.
        if batch_idx % 10 == 0:
            avg_loss = running_loss / (batch_idx + 1)
            print(f"Epoch [{epoch+1}/{constants.NUM_EPOCHS}] Batch [{batch_idx}] Loss: {avg_loss:.4f}")
            writer.add_scalar("Loss/Train/Total", avg_loss, global_step)
            writer.add_scalar("Loss/Train/Regression", reg_loss.item(), global_step)
            writer.add_scalar("Loss/Train/BB_Diff", bb_loss.item(), global_step)
            writer.add_scalar("Loss/Train/FP", false_pos.item(), global_step)
            writer.add_scalar("Loss/Train/FN", false_neg.item(), global_step)
            
            writer.add_scalar("BBTime/Train/Diff", (pred_bbtime - bbtime).abs().mean().item(), global_step)
            
            # Log model parameter statistics every few batches
            if batch_idx % 50 == 0:
                log_model_stats(model, global_step)

    # End-of-epoch training logging
    train_epoch_loss = running_loss / len(train_loader)
    print(f"Epoch [{epoch+1}/{constants.NUM_EPOCHS}] Training Average Loss: {train_epoch_loss:.4f}")
    writer.add_scalar("Loss/Train/Epoch", train_epoch_loss, epoch)
    
    # Validation phase
    val_metrics = evaluate(model, val_loader, constants.DEVICE)
    val_loss = val_metrics["loss"]
    
    # Log validation metrics
    print(f"Epoch [{epoch+1}/{constants.NUM_EPOCHS}] Validation Loss: {val_loss:.4f}")
    writer.add_scalar("Loss/Val/Total", val_loss, epoch)
    writer.add_scalar("Loss/Val/Regression", val_metrics["reg_loss"], epoch)
    writer.add_scalar("Loss/Val/BB_Diff", val_metrics["bb_loss"], epoch)
    writer.add_scalar("Loss/Val/FP", val_metrics["fp_loss"], epoch)
    writer.add_scalar("Loss/Val/FN", val_metrics["fn_loss"], epoch)
    
    # Early stopping and model saving logic
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        
        # Save the best model
        os.makedirs("checkpoints", exist_ok=True)
        best_model_path = 'checkpoints/best_model.pth'
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_epoch_loss,
            'val_loss': val_loss,
        }, best_model_path)
        print(f"Saved best model with validation loss: {val_loss:.4f}")
    else:
        patience_counter += 1
        print(f"Validation loss did not improve. Patience: {patience_counter}/{patience}")
        
        if patience_counter >= patience:
            print(f"Early stopping triggered after {epoch+1} epochs")
            break
    
    # Save model checkpoint for this epoch
    checkpoint_path = f'checkpoints/model_epoch_{epoch+1}.pth'
    torch.save({
        'epoch': epoch + 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': train_epoch_loss,
        'val_loss': val_loss,
    }, checkpoint_path)
    
    # Log full model statistics at the end of each epoch
    log_model_stats(model, global_step)
    
    # Add sample predictions visualization (optional)
    if epoch % 5 == 0 or epoch == constants.NUM_EPOCHS - 1:
        # Get a sample batch for visualization
        sample_batch = next(iter(val_loader))  # Use validation data for consistent samples
        sample_instructions = {k: v.to(constants.DEVICE) for k, v in sample_batch["instructions"].items()}
        sample_positions = sample_batch["positions"].to(constants.DEVICE)
        sample_bbtime = sample_batch["bbtime"].to(constants.DEVICE)
        sample_target = sample_batch["target"].to(constants.DEVICE)
        
        with torch.no_grad():
            sample_preds = model(sample_instructions, sample_positions, sample_bbtime)
        
        # Create text representation for visualization
        for b in range(min(2, sample_preds.size(0))):  # Just show first 2 examples
            sample_text = ""
            for i in range(sample_preds.size(1)):
                if sample_instructions["inst_id"][b, i] == 0:  # Assuming 0 is padding
                    continue
                sample_text += f"Instr {i}: Pred={sample_preds[b, i, 0].item():.4f}, Target={sample_target[b, i, 0].item():.4f}\n"
            writer.add_text(f"Predictions_Example_{b}", sample_text, epoch)
    
# Save the final model.
os.makedirs("checkpoints", exist_ok=True)
torch.save(model.state_dict(), "checkpoints/model_final.pth")

writer.close()
print("Training complete.")
