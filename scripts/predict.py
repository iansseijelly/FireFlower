import os
import sys
import torch
import argparse
import pandas as pd
from tacit_learn.dataloader import BasicBlockDataset, collate_fn
from tacit_learn.model import FireFlowerPredictor, FireFlowerConfig
from math import sqrt
import numpy as np

# =====================
#   Configuration
# =====================
VOCAB_FILE = "vocab/opcodes.txt"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model(model_path):
    """Load the trained model."""
    # Assume we're using the same dataset vocabulary for consistency
    dummy_dataset = BasicBlockDataset(vocab_path=VOCAB_FILE, file_paths=None)
    
    # Initialize model with the same configuration as during training
    NUM_INST = dummy_dataset.get_n_inst()
    NUM_REGS = dummy_dataset.get_n_reg()
    MAX_BLOCK_LEN = 64
    
    config = FireFlowerConfig(
        n_inst=NUM_INST, 
        n_reg=NUM_REGS,
        d_inst=int(sqrt(NUM_INST)),
        d_reg=int(sqrt(NUM_REGS)),
        d_imm=int(sqrt(NUM_INST)),
        d_bb=int(sqrt(NUM_INST)),
        d_model=128,
        n_head=4,
        n_layers=2,
        n_pos=MAX_BLOCK_LEN,
        d_pos=int(sqrt(NUM_REGS))
    )
    
    model = FireFlowerPredictor(config)
    if torch.__version__ >= "2.0.0":
        model = torch.compile(model)
    
    # Load the checkpoint
    checkpoint = torch.load(model_path, map_location=DEVICE)
    
    # Check if it's a training checkpoint or just model weights
    model.load_state_dict(checkpoint["model_state_dict"])
    
    model.to(DEVICE)
    model.eval()
    
    return model, dummy_dataset

def predict_basic_block(model, dataset, file_path=None, input_text=None):
    """
    Predict latencies for a basic block.
    
    Args:
        model: The trained FireFlowerPredictor model
        dataset: The BasicBlockDataset for handling input processing
        file_paths: Path to a file containing basic block(s)
        input_text: Direct text input of basic block(s)
    
    Returns:
        A list of predictions with instruction details
    """
    if file_path is None and input_text is None:
        raise ValueError("Either file_paths or input_text must be provided")
    
    # Create a temporary file if input_text is provided
    if input_text is not None:
        import tempfile
        temp = tempfile.NamedTemporaryFile(mode='w+', delete=False)
        temp.write(input_text)
        temp.close()
        file_path = temp.name
    
    # Load the data
    try:
        test_dataset = BasicBlockDataset(vocab_path=VOCAB_FILE, file_paths=[file_path])
        
        # Process all basic blocks in the file
        results = []
        
        with torch.no_grad():
            for idx in range(len(test_dataset)):
                # Get a single basic block
                sample = test_dataset[idx]
                
                # Prepare batch (a single basic block)
                batch = collate_fn([sample])
                
                # Move to device
                instructions = {k: v.to(DEVICE) for k, v in batch["instructions"].items()}
                positions = batch["positions"].to(DEVICE)
                bbtime = batch["bbtime"].float().to(DEVICE)
                target = batch["target"].float().to(DEVICE)
                
                # Forward pass
                preds = model(instructions, positions, bbtime)
                
                # Extract original instruction text and other details
                instr_text = batch["instruction_text"][0]  # Get first (and only) item in the batch
                instr_types = [test_dataset.id2inst.get(i.item(), "unknown") for i in instructions["inst_id"][0] if i.item() != 0]
                
                # Calculate total block time
                total_pred = preds.sum().item()
                total_target = bbtime.item()
                
                # Format results for this basic block
                block_results = {
                    "block_idx": idx,
                    "total_predicted": total_pred,
                    "total_actual": total_target,
                    "instructions": []
                }
                
                # Add details for each instruction
                for i in range(len(instr_text)):
                    if i < preds.shape[1]:  # Make sure we have a prediction for this instruction
                        block_results["instructions"].append({
                            "instruction": instr_text[i],
                            "opcode": instr_types[i],
                            "predicted_latency": preds[0, i, 0].item(),
                            "actual_latency": target[0, i, 0].item() if i < target.shape[1] else None,
                            "position": i
                        })
                
                results.append(block_results)
        
        # Clean up temporary file if we created one
        if input_text is not None:
            os.unlink(file_path)
            
        return results
    
    except Exception as e:
        # Clean up temporary file if we created one and an error occurred
        if input_text is not None and 'file_paths' in locals():
            os.unlink(file_path)
        raise e

def format_results(results):
    """Format the prediction results in a human-readable way."""
    output = []
    
    for block in results:
        output.append(f"\n===== Basic Block {block['block_idx']} =====")
        output.append(f"Total Predicted Time: {block['total_predicted']:.4f}")
        output.append(f"Total Actual Time: {block['total_actual']:.4f}")
        output.append(f"Difference: {block['total_predicted'] - block['total_actual']:.4f}\n")
        
        output.append(f"Target BB time: {block['total_actual']:.4f}")
        # Table header
        output.append(f"{'Position':<8} {'Opcode':<15} {'Predicted':<12} {'Actual':<12} {'Instruction'}")
        output.append("-" * 80)
        # target bbtime
        # Table rows
        for instr in block["instructions"]:
            actual = f"{instr['actual_latency']:.4f}" if instr['actual_latency'] is not None else "N/A"
            output.append(
                f"{instr['position']:<8} {instr['opcode'][:15]:<15} {instr['predicted_latency']:.4f}       "
                f"{actual:<12} {instr['instruction']}"
            )
            
        # Add separator between blocks
        output.append("\n" + "=" * 80)
    
    return "\n".join(output)

def main():
    parser = argparse.ArgumentParser(description="Predict instruction latencies in basic blocks")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("-f", "--file", help="Path to file containing basic block(s)")
    group.add_argument("-t", "--text", help="Basic block text directly as input")
    parser.add_argument("-m", "--model", help="Path to model checkpoint")
    parser.add_argument("-o", "--output", help="Output file path (default: output to console)")
    
    args = parser.parse_args()
    
    try:
        # Load model
        print("Loading model...")
        if args.model is None:
            MODEL_PATH = "checkpoints/model_final.pth"
        else:
            MODEL_PATH = args.model
        model, dataset = load_model(MODEL_PATH)
        
        # Make predictions
        print("Running predictions...")
        results = predict_basic_block(model, dataset, 
                                      file_path=args.file, 
                                      input_text=args.text)
        
        # Format results
        formatted_output = format_results(results)
        
        # Output results
        if args.output:
            with open(args.output, 'w') as f:
                f.write(formatted_output)
            print(f"Results saved to {args.output}")
        else:
            print(formatted_output)
            
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
        
    return 0

if __name__ == "__main__":
    sys.exit(main()) 