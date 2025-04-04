import torch
import random
from tqdm.auto import tqdm
import numpy as np
import os
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from transformers import BertForMaskedLM, BertConfig
from torch.utils.data import DataLoader, Dataset

from tacit_learn.tokenizer import Tokenizer



# training_data = "./data/dummy.txt"
training_data = "./data/baremetal_startup.txt"

# Initialize tokenizer and model
tokenizer = Tokenizer()

# Create custom config with actual vocab size
config = BertConfig(
    vocab_size=tokenizer.num_tokens,
    hidden_size=64,
    num_hidden_layers=8,
    num_attention_heads=4,
    intermediate_size=512
)

model = BertForMaskedLM(config)


num_epochs = 40



# Set seed for reproducibility
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Create TensorBoard writer
log_dir = os.path.join("logs", datetime.now().strftime("%Y%m%d-%H%M%S"))
writer = SummaryWriter(log_dir)
print(f"TensorBoard logs will be saved to {log_dir}")

# Load training data
print(f"Loading training data from {training_data}")
with open(training_data, "r") as f:
    training_data = f.read().strip().split("\n")

print(f"Loaded {len(training_data)} lines of training data")


# Still randomize if training from scratch
for param in model.parameters():
    param.data[:] = torch.randn_like(param.data)


# Create masked input dataset
class MaskingDataset(Dataset):
    def __init__(self, examples, tokenizer, mask_probability=0.15):
        self.examples = examples
        self.tokenizer = tokenizer
        self.mask_probability = mask_probability
        
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        # Tokenize the example
        tokens = self.examples[idx].split()
        
        # Choose positions to mask (excluding special markers like START, END)
        maskable_positions = [i for i, token in enumerate(tokens) 
                             if token not in ["START", "END", "TIMESTAMP"]]
        
        # Skip if no maskable positions
        if not maskable_positions:
            raise Exception("No maskable positions found")
            return self.tokenizer(self.examples[idx])
        
        # Create a masked version (for model input)
        masked_tokens = tokens.copy()
        num_to_mask = max(1, int(len(maskable_positions) * self.mask_probability))
        mask_positions = random.sample(maskable_positions, num_to_mask)
        
        for pos in mask_positions:
            masked_tokens[pos] = "[MASK]"
        
        # Create input and label
        masked_text = " ".join(masked_tokens)
        original_text = " ".join(tokens)
        
        # Tokenize both versions
        masked_encoding = self.tokenizer(masked_text)
        
        label_encoding = self.tokenizer(original_text)
        
        # Prepare final encoding
        encoding = {
            "input_ids": masked_encoding["input_ids"][0],
            "token_type_ids": masked_encoding["token_type_ids"][0],
            "attention_mask": masked_encoding["attention_mask"][0],
            "labels": label_encoding["input_ids"][0],
        }
        
        return encoding


# Create dataset and dataloader
train_dataset = MaskingDataset(training_data, tokenizer)
train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)
num_training_data = len(train_dataset)
print(f"Created dataset with {num_training_data} lines of training data")

# Initialize optimizer and scheduler
from torch.optim import AdamW

optimizer = AdamW(model.parameters(), lr=1e-4)

from transformers import get_scheduler

num_training_steps = num_epochs * len(train_dataloader)
lr_scheduler = get_scheduler(
    name="linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
)

# Ensure model is on the right device
model.to(device)

# Define example inputs for tracking progress throughout training
example_inputs = [
    "x0 x0 [MASK] x0 x0 x0 x0 x0 x0 x0 x0 x0 x0 x0 x0",
    "START INST addi [MASK] x3 RS1 x0 IMM 0 TIMESTAMP 0 END",
    "START INST sw RS1 x8 RS2 x4 IMM 0 TIMESTAMP [MASK] END",
]

def predict_masked_token(model, tokenizer, text):
    # Encode the input
    encoded_input = tokenizer(text)
    encoded_input = {k: v.to(device) for k, v in encoded_input.items()}
    
    # Find mask token positions
    mask_positions = (encoded_input["input_ids"] == tokenizer.mask_token_id).nonzero(as_tuple=True)
    
    if mask_positions[0].shape[0] == 0:
        print("No mask token found in the input!")
        return None
    
    # Generate predictions
    with torch.no_grad():
        outputs = model(**encoded_input)
    
    # Get predictions for all mask positions
    results = []
    for batch_idx, pos in zip(mask_positions[0], mask_positions[1]):
        # Get top 5 predictions
        logits = outputs.logits[batch_idx, pos, :]
        top_5_tokens = torch.topk(logits, 5, dim=0)
        top_token_ids = top_5_tokens.indices.tolist()
        top_token_probs = torch.softmax(top_5_tokens.values, dim=0).tolist()
        
        # Convert token IDs to words
        top_tokens = [tokenizer.decode([token_id]) for token_id in top_token_ids]
        
        results.append({
            "position": pos.item(),
            "top_predictions": [
                {"token": token, "probability": prob} 
                for token, prob in zip(top_tokens, top_token_probs)
            ]
        })
    
    return results

def log_example_predictions(epoch):
    """Run example predictions and log to TensorBoard"""
    model.eval()
    
    # Log example predictions during training
    for idx, example in enumerate(example_inputs):
        predictions = predict_masked_token(model, tokenizer, example)
        if not predictions:
            continue
            
        # Log the top prediction for each mask
        for p_idx, p in enumerate(predictions):
            if p["top_predictions"]:
                top_pred = p["top_predictions"][0]
                writer.add_text(
                    f"Example {idx+1}/Position {p['position']}", 
                    f"Epoch {epoch}: '{top_pred['token']}' (p={top_pred['probability']:.4f})",
                    epoch
                )
                
                # Also add a scalar for the top prediction confidence
                writer.add_scalar(
                    f"Confidence/Example_{idx+1}_Pos_{p['position']}", 
                    top_pred['probability'],
                    epoch
                )
                
                # Log top 5 predictions as histogram
                probs = [pred["probability"] for pred in p["top_predictions"]]
                writer.add_histogram(
                    f"Top5Probs/Example_{idx+1}_Pos_{p['position']}", 
                    torch.tensor(probs), 
                    epoch
                )
    
    model.train()





# Training loop
progress_bar = tqdm(range(num_training_steps))
step_counter = 0


# Run example predictions and log to TensorBoard
log_example_predictions(0)

# Add model parameter histograms to TensorBoard
for name, param in model.named_parameters():
    if param.requires_grad:
        writer.add_histogram(f"Parameters/{name}", param.data, 0)

# Log learning rate
writer.add_scalar('LearningRate', optimizer.param_groups[0]['lr'], 0)


for epoch in range(num_epochs):
    epoch_loss = 0.0
    for batch in train_dataloader:
        # Move batch to device
        batch = {k: v.to(device) for k, v in batch.items()}
        
        # Forward pass
        outputs = model(**batch)
        loss = outputs.loss

        # Backward pass
        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        
        # Update progress and report
        progress_bar.update(1)
        step_counter += 1
        epoch_loss += loss.item()
        
        # Log to TensorBoard
        writer.add_scalar('Loss/train_step', loss.item(), step_counter)
        
        print(f"Step {step_counter}, Loss: {loss.item():.4f}")
    
    # Calculate and report epoch statistics
    avg_epoch_loss = epoch_loss / len(train_dataloader)
    writer.add_scalar('Loss/train_epoch', avg_epoch_loss, epoch)
    print(f"\nEpoch {epoch+1}/{num_epochs} completed. Average Loss: {avg_epoch_loss:.4f}")
    
    # Run example predictions and log to TensorBoard
    log_example_predictions(epoch)
    
    # Add model parameter histograms to TensorBoard
    for name, param in model.named_parameters():
        if param.requires_grad:
            writer.add_histogram(f"Parameters/{name}", param.data, epoch)
            
    # Log learning rate
    writer.add_scalar('LearningRate', optimizer.param_groups[0]['lr'], epoch)

# Close TensorBoard writer
writer.close()


# Save the model
print("Saving model...")
model.save_pretrained("trained_model")
tokenizer.save_pretrained("trained_model")
print("Model saved!")



# Evaluate model on test examples
print("\n=== Testing the model on example inputs ===")

# Example 1: Predict instruction operand
example1 = "START INST addi RD x1 RS1 [MASK] IMM 0 TIMESTAMP 0 END"
print(f"\nExample 1: {example1}")
predictions = predict_masked_token(model, tokenizer, example1)
if predictions:
    for p in predictions:
        print(f"Position {p['position']}:")
        for pred in p["top_predictions"]:
            print(f"  {pred['token']} (prob: {pred['probability']:.4f})")

# Example 2: Predict instruction type
example2 = "START INST [MASK] RD x2 RS1 x0 IMM 10 TIMESTAMP 0 END"
print(f"\nExample 2: {example2}")
predictions = predict_masked_token(model, tokenizer, example2)
if predictions:
    for p in predictions:
        print(f"Position {p['position']}:")
        for pred in p["top_predictions"]:
            print(f"  {pred['token']} (prob: {pred['probability']:.4f})")

# Example 3: Predict register
example3 = "START INST sw RS1 x8 RS2 [MASK] IMM 0 TIMESTAMP 0 END"
print(f"\nExample 3: {example3}")
predictions = predict_masked_token(model, tokenizer, example3)
if predictions:
    for p in predictions:
        print(f"Position {p['position']}:")
        for pred in p["top_predictions"]:
            print(f"  {pred['token']} (prob: {pred['probability']:.4f})")
