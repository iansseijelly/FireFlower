import os
import random
from datetime import datetime

import numpy as np
from tqdm.auto import tqdm
import torch
from torch.utils.tensorboard import SummaryWriter
from transformers import BertForMaskedLM, BertConfig
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from transformers import get_scheduler
from tacit_learn.tokenizer import Tokenizer



# training_data = "./data/dummy.txt"
training_data = "./data/rocket-hello.canonicalized.out"

# Initialize tokenizer and model
tokenizer = Tokenizer(vocab_file="./vocab/riscv_vocab.txt")

# Create custom config with actual vocab size
config = BertConfig(
    vocab_size=tokenizer.num_tokens,
    hidden_size=256,
    num_hidden_layers=8,
    num_attention_heads=8,
    intermediate_size=1024
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
    training_data = f.read().strip().split("ENDBB")[:-1]

print(f"Loaded {len(training_data)} lines of training data")

# tokenize training data
training_inputs = tokenizer(training_data, max_length=200)

# create a clone of the input ids as ground truth labels
training_inputs["labels"] = training_inputs["input_ids"].detach().clone()

# choose one position to mask (excluding special markers like SEP, CLS, PAD)
for i in range(training_inputs["input_ids"].shape[0]):
    maskable_positions = [j for j, token in enumerate(training_inputs["input_ids"][i])
                          if token not in [tokenizer.sep_token_id, tokenizer.cls_token_id, tokenizer.pad_token_id]]
    if not maskable_positions:
        raise Exception("No maskable positions found")
    
    mask_position = random.choice(maskable_positions)

    # print the maskable positions
    print(f"Maskable positions: {maskable_positions}")
    print(f"Mask position: {mask_position}")
    print(f"Masked token: {tokenizer.decode(training_inputs['input_ids'][i, mask_position])}")
    print(f"data before masking: {training_data[i]}")
    # breakpoint()

    training_inputs["input_ids"][i, mask_position] = tokenizer.mask_token_id


# make masks after every TIMESTAMP token
# for i in range(training_inputs["input_ids"].shape[0]):
#     for j in range(training_inputs["input_ids"].shape[1]):
#         if tokenizer.decode(training_inputs["input_ids"][i, j]) == "TIMESTAMP":
#             training_inputs["input_ids"][i, j + 1] = tokenizer.mask_token_id


class TraceDataset(Dataset):
    def __init__(self, training_inputs):
        assert training_inputs["input_ids"].dim() == 2
        assert training_inputs["input_ids"].shape == training_inputs["labels"].shape
        assert training_inputs["input_ids"].shape == training_inputs["attention_mask"].shape
        assert training_inputs["input_ids"].shape == training_inputs["token_type_ids"].shape
        
        self.training_inputs = training_inputs

    def __len__(self) -> int:
        return self.training_inputs["input_ids"].shape[0]

    def __getitem__(self, index) -> dict:
        input_ids = self.training_inputs["input_ids"][index]
        attention_mask = self.training_inputs["attention_mask"][index]
        token_type_ids = self.training_inputs["token_type_ids"][index]
        labels = self.training_inputs["labels"][index]

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
            "labels": labels,
        }


# create dataset and dataloader
train_dataset = TraceDataset(training_inputs)
train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)

num_training_data = len(train_dataset)
print(f"Created dataset with {num_training_data} lines of training data")

# Initialize optimizer and scheduler


optimizer = AdamW(model.parameters(), lr=1e-4)

num_training_steps = num_epochs * len(train_dataloader)
lr_scheduler = get_scheduler(
    name="linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
)

# ensure model is on the right device
model.to(device)

# define example inputs for tracking progress throughout training
example_inputs = [
    "x0 x0 [MASK] x0 x0 x0 x0 x0 x0 x0 x0 x0 x0 x0 x0",
    "START INST addi [MASK] x3 RS1 x0 IMM 0 TIMESTAMP 0 END",
    "START INST sw RS1 x8 RS2 x4 IMM 0 TIMESTAMP [MASK] END",
]

def predict_masked_token(model, tokenizer, text):
    # encode the input
    encoded_input = tokenizer(text)
    encoded_input = {k: v.to(device) for k, v in encoded_input.items()}
    
    # find mask token positions
    mask_positions = (encoded_input["input_ids"] == tokenizer.mask_token_id).nonzero(as_tuple=True)
    
    if mask_positions[0].shape[0] == 0:
        raise Exception("No mask token found in the input!")
    
    # generate predictions
    with torch.no_grad():
        outputs = model(**encoded_input)
    
    
    # get top 5 predictions
    logits = outputs.logits[mask_positions[0], mask_positions[1]][0]
    top_5_tokens = torch.topk(logits, 5, dim=0)
    top_token_ids = top_5_tokens.indices.tolist()
    top_token_probs = torch.softmax(top_5_tokens.values, dim=0).tolist()
        
    # Convert token IDs to words
    top_tokens = [tokenizer.decode([token_id]) for token_id in top_token_ids]

    result = {
        "position": mask_positions[0].item(),
        "top_predictions": [
            {"token": token, "probability": prob} 
            for token, prob in zip(top_tokens, top_token_probs)
        ]
    }
    
    return result

def log_example_predictions(epoch):
    """Run example predictions and log to TensorBoard"""
    model.eval()
    
    # Log example predictions during training
    for idx, example in enumerate(example_inputs):
        prediction = predict_masked_token(model, tokenizer, example)
        
        # Log the top prediction for each mask
        if prediction["top_predictions"]:
            top_pred = prediction["top_predictions"][0]
            writer.add_text(
                f"Example {idx+1}/Position {prediction['position']}", 
                f"Epoch {epoch}: '{top_pred['token']}' (p={top_pred['probability']:.4f})",
                epoch
            )
            
            # Also add a scalar for the top prediction confidence
            writer.add_scalar(
                f"Confidence/Example_{idx+1}_Pos_{prediction['position']}", 
                top_pred['probability'],
                epoch
            )
            
            # Log top 5 predictions as histogram
            probs = [pred["probability"] for pred in prediction["top_predictions"]]
            writer.add_histogram(
                f"Top5Probs/Example_{idx+1}_Pos_{prediction['position']}", 
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
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        token_type_ids = batch["token_type_ids"].to(device)
        labels = batch["labels"].to(device)
        
        # Forward pass
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )

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
example1 = '''START PC 800001bc INST sd RS1 x5 RS2 x0 IMM 0 TIMESTAMP [MASK] END
START PC 800001c0 INST addi RD x5 RS1 x5 IMM 8 TIMESTAMP 1 END
START PC 800001c4 INST bltu RS1 x5 RS2 x6 IMM -8 TIMESTAMP 1 END
BBTIME 3'''
print(f"\nExample 1: {example1}")
predictions = predict_masked_token(model, tokenizer, example1)
print(f"Position {predictions['position']}:")
for pred in predictions["top_predictions"]:
    print(f"  {pred['token']} (prob: {pred['probability']:.4f})")

# Example 2: Predict instruction type
# Example 1: Predict instruction operand
example2 = '''START PC 800001bc INST sd RS1 x5 RS2 x0 IMM 0 TIMESTAMP 1 END
START PC 800001c0 INST addi RD x5 RS1 x5 IMM 8 TIMESTAMP [MASK] END
START PC 800001c4 INST bltu RS1 x5 RS2 x6 IMM -8 TIMESTAMP 1 END
BBTIME 45'''
print(f"\nExample 2: {example2}")
predictions = predict_masked_token(model, tokenizer, example2)
print(f"Position {predictions['position']}:")
for pred in predictions["top_predictions"]:
    print(f"  {pred['token']} (prob: {pred['probability']:.4f})")

# Example 3: Predict something simple
example3 = "START PC 800001c0 INST [MASK] RD x5 RS1 x5 IMM 8 TIMESTAMP 43 END"
print(f"\nExample 3: {example3}")
predictions = predict_masked_token(model, tokenizer, example3)
print(f"Position {predictions['position']}:")
for pred in predictions["top_predictions"]:
    print(f"  {pred['token']} (prob: {pred['probability']:.4f})")
