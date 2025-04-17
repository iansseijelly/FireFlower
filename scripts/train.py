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
from tacit_learn.dataloader import BasicBlockDataset, collate_fn
from tacit_learn.model import FireFlowerPredictor, FireFlowerConfig

# =====================
#   Hyperparameters
# =====================
DATA_FOLDER = "data/canonicalized"  # Your dataset file path.
VOCAB_FILE = "vocab/opcodes.txt"
BATCH_SIZE = 64
NUM_EPOCHS = 10
LEARNING_RATE = 1e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LOG_DIR = "./logs/ff_training_" + datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

# Weight for the block-level sum constraint.
BB_LOSS_WEIGHT = 0.1

# Configuration
max_checkpoints_to_keep = 5  # Adjust as needed

# =====================
#   Dataset & DataLoader
# =====================
training_files = glob.glob(os.path.join(DATA_FOLDER, "*.out"))
train_dataset = BasicBlockDataset(vocab_path=VOCAB_FILE, file_paths=training_files)
train_loader = DataLoader(train_dataset, 
                        batch_size=BATCH_SIZE, 
                        collate_fn=collate_fn, 
                        shuffle=True,
                        pin_memory=True,
                        num_workers=4)


# =====================
#   Instantiate the Model
# =====================
# Adjust the vocabulary sizes as needed.
NUM_INST = train_dataset.get_n_inst()
NUM_REGS = train_dataset.get_n_reg()
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

if torch.__version__ >= "2.0.0":
    model = torch.compile(model)
else:
    print("Warning: torch version is less than 2.0.0, model will not be compiled.")

model.to(DEVICE)


# =====================
#   Loss & Optimizer
# =====================
mse_loss = nn.MSELoss()
optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
max_grad_norm = 1.0

# =====================
#   TensorBoard Setup
# =====================
writer = SummaryWriter(LOG_DIR)

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
#   Training Loop
# =====================
global_step = 0

for epoch in range(NUM_EPOCHS):
    model.train()
    running_loss = 0.0
    # Log learning rate at start of epoch
    writer.add_scalar("Learning_Rate", optimizer.param_groups[0]['lr'], epoch)
    
    for batch_idx, batch in enumerate(train_loader):
        # Move batch to device.
        instructions = {k: v.to(DEVICE) for k, v in batch["instructions"].items()}
        # print(f"Max inst_id: {instructions['inst_id'].max().item()}, vocab size: {config.n_inst}")
        # print(f"Max position: {batch['positions'].max().item()}, max allowed: {config.n_pos - 1}")
        positions = batch["positions"].to(DEVICE)
        bbtime = batch["bbtime"].to(DEVICE)  # shape: [B, 1]
        # Our target is the TIMESTAMP field per instruction.
        target = batch["target"].to(DEVICE)   # shape: [B, L, 1]

        optimizer.zero_grad()

        # Forward pass.
        preds = model(instructions, positions, bbtime)  # shape: [B, L, 1]

        # Compute per-instruction regression loss.
        reg_loss = mse_loss(preds, target)

        # Compute block-level sum loss.
        # Sum predictions over instructions (dim=1) and compare with bbtime.
        # Note: our target for bbtime is a scalar per block.
        pred_bbtime = preds.sum(dim=1)  # shape: [B, 1]
        bb_loss = mse_loss(pred_bbtime, bbtime)

        # After computing losses
        if torch.isnan(reg_loss) or torch.isnan(bb_loss):
            print(f"NaN detected! reg_loss: {reg_loss.item()}, bb_loss: {bb_loss.item()}")
            print(f"preds min/max: {preds.min().item()}/{preds.max().item()}")
            print(f"target min/max: {target.min().item()}/{target.max().item()}")
            print(f"bbtime min/max: {bbtime.min().item()}/{bbtime.max().item()}")
            # Optional: skip this batch
            breakpoint()

        loss = reg_loss + BB_LOSS_WEIGHT * bb_loss

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
            print(f"Epoch [{epoch+1}/{NUM_EPOCHS}] Batch [{batch_idx}] Loss: {avg_loss:.4f}")
            writer.add_scalar("Loss/Total", avg_loss, global_step)
            writer.add_scalar("Loss/Regression", reg_loss.item(), global_step)
            writer.add_scalar("Loss/BB_Sum", bb_loss.item(), global_step)
            
            # Log prediction and target statistics
            writer.add_scalar("Predictions/Min", preds.min().item(), global_step)
            writer.add_scalar("Predictions/Max", preds.max().item(), global_step)
            writer.add_scalar("Predictions/Mean", preds.mean().item(), global_step)
            writer.add_scalar("Predictions/Std", preds.std().item(), global_step)
            
            writer.add_scalar("Targets/Min", target.min().item(), global_step)
            writer.add_scalar("Targets/Max", target.max().item(), global_step)
            writer.add_scalar("Targets/Mean", target.mean().item(), global_step)
            writer.add_scalar("Targets/Std", target.std().item(), global_step)
            
            writer.add_scalar("BBTime/Min", bbtime.min().item(), global_step)
            writer.add_scalar("BBTime/Max", bbtime.max().item(), global_step)
            writer.add_scalar("BBTime/Mean", bbtime.mean().item(), global_step)
            
            # Log prediction sum vs actual bbtime
            writer.add_scalar("BBTime/PredictionSum", pred_bbtime.mean().item(), global_step)
            writer.add_scalar("BBTime/Actual", bbtime.mean().item(), global_step)
            writer.add_scalar("BBTime/Diff", (pred_bbtime - bbtime).abs().mean().item(), global_step)
            
            # Log model parameter statistics every few batches
            if batch_idx % 50 == 0:
                log_model_stats(model, global_step)

    # End-of-epoch logging.
    epoch_loss = running_loss / len(train_loader)
    print(f"Epoch [{epoch+1}/{NUM_EPOCHS}] Average Loss: {epoch_loss:.4f}")
    writer.add_scalar("Loss/Epoch", epoch_loss, epoch)
    
    # Log full model statistics at the end of each epoch
    log_model_stats(model, global_step)
    
    # Add sample predictions visualization (optional)
    if epoch % 5 == 0 or epoch == NUM_EPOCHS - 1:
        # Get a sample batch for visualization
        sample_batch = next(iter(train_loader))
        sample_instructions = {k: v.to(DEVICE) for k, v in sample_batch["instructions"].items()}
        sample_positions = sample_batch["positions"].to(DEVICE)
        sample_bbtime = sample_batch["bbtime"].to(DEVICE)
        sample_target = sample_batch["target"].to(DEVICE)
        
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

    # Save model after every epoch
    checkpoint_path = f'checkpoints/model_epoch_{epoch+1}.pth'
    torch.save({
        'epoch': epoch + 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': epoch_loss,
    }, checkpoint_path)
    
    # Optional: Remove old checkpoints to save space
    if max_checkpoints_to_keep > 0:
        checkpoint_files = sorted([f for f in os.listdir('checkpoints') 
                                  if f.startswith('model_epoch_')])
        while len(checkpoint_files) > max_checkpoints_to_keep:
            os.remove(os.path.join('checkpoints', checkpoint_files[0]))
            checkpoint_files.pop(0)

# Save the final model.
os.makedirs("checkpoints", exist_ok=True)
torch.save(model.state_dict(), "checkpoints/model_final.pth")

writer.close()
print("Training complete.")
