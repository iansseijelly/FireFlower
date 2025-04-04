import torch
import os
from transformers import BertForMaskedLM
from tacit_learn.tokenizer import Tokenizer


# Set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# Path to the saved model checkpoint
model_path = "trained_model"

# Load tokenizer and model from checkpoint
tokenizer = Tokenizer()
model = BertForMaskedLM.from_pretrained(model_path).to(device)
model.eval()

print(f"Loaded model from {model_path}")


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

# Define test examples
test_examples = [
    "x0 x0 [MASK] x0 x0 x0 x0 x0 x0 x0 x0 x0 x0 x0 x0",
    "START INST addi [MASK] x3 RS1 x0 IMM 0 TIMESTAMP 0 END",
    "START INST sw RS1 x8 RS2 x4 IMM 0 TIMESTAMP [MASK] END",
    # Add more examples here as needed
]

# Run predictions on each example
print("\n=== Running mask filling predictions ===")

for i, example in enumerate(test_examples):
    print(f"\nExample {i+1}: {example}")
    predictions = predict_masked_token(model, tokenizer, example)
    
    if predictions:
        print(f"Position {predictions['position']}:")
        for pred in predictions["top_predictions"]:
            print(f"  {pred['token']} (prob: {pred['probability']:.4f})")
    else:
        print("No predictions generated.")
