import torch
from torch.utils.data import Dataset, DataLoader
import re

# A helper function to parse a single instruction line.
def parse_instruction_line(line):
    """
    Example instruction line:
    START PC 800001bc INST sd RS1 x5 RS2 x0 IMM 0 TIMESTAMP 1 END
    We extract:
      - inst_type (from the token after "INST")
      - rd (from token after "RD")
      - rs1 (from token after "RS1")
      - rs2 (from token after "RS2")
      - imm (from token after "IMM") as a float
      - timestamp (from token after "TIMESTAMP") as a float
    (For simplicity we assume these fields exist; in practice, you'd add error checking.)
    """
    tokens = line.strip().split()
    # We'll use regex or token-by-token search. Here, we simply iterate and look for known keywords.
    fields = {}
    i = 0
    while i < len(tokens):
        token = tokens[i]
        if token == "INST":
            fields["inst_type"] = tokens[i+1]
            i += 2
        elif token == "RD":
            fields["rd"] = tokens[i+1]
            i += 2
        elif token in ["RS1"]:  # sometimes the first register may be labeled as RD
            fields["rs1"] = tokens[i+1]
            i += 2
        elif token == "RS2":
            fields["rs2"] = tokens[i+1]
            i += 2
        elif token == "IMM":
            # Convert immediate to float.
            fields["imm"] = float(tokens[i+1])
            i += 2
        elif token == "TIMESTAMP":
            # This is the target latency for the instruction.
            fields["timestamp"] = float(tokens[i+1])
            i += 2
        else:
            i += 1
    return fields

class BasicBlockDataset(Dataset):
    def __init__(self, vocab_path, file_paths: list | None = None, max_block_len=64):
        """
        Reads the file and parses basic blocks.
        Each basic block ends with a line like "BBTIME <value> ENDBB".
        """
        self.blocks = []
        self.max_block_len = max_block_len

        # Load vocabulary
        self.load_vocab(vocab_path)
        
        # If file_path is None, just initialize empty data structures
        if file_paths is None:
            self.data = []
            return
        
        # Otherwise load the data
        for file_path in file_paths:
            self.load_data(file_path)

    def load_vocab(self, vocab_path):
        # load the opcode vocab
        with open(vocab_path, "r") as f:
            vocab = f.read().strip().splitlines()
        self.inst_vocab = {inst: i+1 for i, inst in enumerate(vocab)}
        self.id2inst = {i+1: inst for i, inst in enumerate(vocab)}  # Add reverse mapping

        # produce the register vocab
        self.rreg_vocab = {f"x{i}": i for i in range(32)}
        self.freg_vocab = {f"f{i}": i+32 for i in range(32)}
        self.reg_vocab = {**self.rreg_vocab, **self.freg_vocab} # concatenate the two vocabs
        self.id2reg = {i: reg for reg, i in self.reg_vocab.items()}  # Add reverse mapping

    def load_data(self, file_path):
        # Read the file.
        with open(file_path, "r") as f:
            lines = f.read().strip().splitlines()

        # Temporary list to collect lines for current basic block
        current_block = []
        for line in lines:
            line = line.strip()
            if line.startswith("BBTIME"):
                # This line contains the basic block delay and ends the block
                # Example: "BBTIME 3 ENDBB"
                parts = line.split()
                # Assuming the format is "BBTIME <value> ENDBB"
                bbtime = float(parts[1])
                # Save the block (instructions and bbtime)
                self.blocks.append((current_block.copy(), bbtime))
                # Reset current block for next basic block
                current_block = []
            elif line:
                # Otherwise, it's an instruction line
                current_block.append(line)

    def get_n_inst(self):
        return len(self.inst_vocab) + 1 

    def get_n_reg(self):
        return len(self.reg_vocab) + 1 

    def __len__(self):
        return len(self.blocks)

    def __getitem__(self, idx):
        """
        Return a single basic block example in dictionary form.
        We'll create:
          - instructions: a dict with keys:
                "inst_id": LongTensor [L] (instruction type indices)
                "rd_id": LongTensor [L] (for RD)
                "rs1_id": LongTensor [L] (for RS1)
                "rs2_id": LongTensor [L] (for RS2; pad if missing)
                "imm": FloatTensor [L, 1]
                "timestamp": LongTensor [L, 1] (the target per-instruction latency)
          - positions: LongTensor [L] with instruction positions 0, 1, ... L-1
          - bbtime: LongTensor [1] overall delay for the block.
          - instruction_text: List of original instruction text lines
        """
        instruction_lines, bbtime = self.blocks[idx]
        
        inst_ids = []
        rd_ids = []
        rs1_ids = []
        rs2_ids = []
        imms = []
        timestamps = []
        
        for line in instruction_lines:
            fields = parse_instruction_line(line)
            # Lookup instruction type. Use 0 if not found (as padding/unknown).
            inst_ids.append(self.inst_vocab.get(fields.get("inst_type", ""), 0))
            # For RD
            rd_ids.append(self.reg_vocab.get(fields.get("rd", ""), 0))
            # For RS1
            rs1_ids.append(self.reg_vocab.get(fields.get("rs1", ""), 0))
            # For RS2: if missing, use 0.
            rs2_ids.append(self.reg_vocab.get(fields.get("rs2", ""), 0))
            imms.append([fields.get("imm", 0.0)])  # wrap in list for shape consistency.
            timestamps.append([fields.get("timestamp", 0.0)])
        
        # Compute positions: simply 0, 1, 2, ... for the block.
        positions = list(range(len(inst_ids)))
        
        # Convert to tensors.
        instructions = {
            "inst_id": torch.tensor(inst_ids, dtype=torch.long),
            "rd_id": torch.tensor(rd_ids, dtype=torch.long),
            "rs1_id": torch.tensor(rs1_ids, dtype=torch.long),
            "rs2_id": torch.tensor(rs2_ids, dtype=torch.long),
            "imm": torch.tensor(imms, dtype=torch.float),           # shape: [L, 1]
            "timestamp": torch.tensor(timestamps, dtype=torch.float)  # shape: [L, 1] target
        }
        positions = torch.tensor(positions, dtype=torch.long)
        bbtime = torch.tensor([bbtime], dtype=torch.float)  # shape: [1]

        sample = {
            "instructions": instructions,
            "positions": positions,
            "bbtime": bbtime,
            "target": instructions["timestamp"],
            "instruction_text": instruction_lines  # Add the original instruction text
        }
        return sample


def collate_fn(batch):
    """
    Collate function to pad basic blocks in the batch to the same length.
    Assumes each item in batch is a dict as returned by BasicBlockDataset.__getitem__.
    """
    batch_size = len(batch)
    # Determine maximum sequence length in the batch.
    max_len = max(item["positions"].size(0) for item in batch)
    
    def pad_tensor(tensor, pad_value=0):
        # tensor shape is [L, ...]
        L = tensor.size(0)
        if L < max_len:
            # Create a padded tensor of shape [max_len, ...].
            pad_shape = (max_len - L, ) + tensor.size()[1:]
            padded = torch.full(pad_shape, pad_value, dtype=tensor.dtype)
            tensor = torch.cat([tensor, padded], dim=0)
        return tensor

    # Prepare batched tensors.
    batch_instructions = {}
    for key in batch[0]["instructions"]:
        # Pad each field along the sequence dimension (dim=0).
        batch_instructions[key] = torch.stack([pad_tensor(item["instructions"][key]) for item in batch], dim=0)
    batch_positions = torch.stack([pad_tensor(item["positions"]) for item in batch], dim=0)
    batch_bbtime = torch.stack([item["bbtime"] for item in batch], dim=0)
    # Target can be part of instructions, e.g., instructions["timestamp"]
    batch_targets = batch_instructions["timestamp"]
    
    # Create padding mask: 1 for real instructions, 0 for padding
    # Shape: [batch_size, max_len, 1]
    padding_mask = (batch_instructions["inst_id"] != 0).unsqueeze(-1).float()
    
    # Collect instruction text (no need to pad as this is just a list)
    batch_instruction_text = [item["instruction_text"] for item in batch]

    return {
        "instructions": batch_instructions,
        "positions": batch_positions,
        "bbtime": batch_bbtime,
        "target": batch_targets,
        "instruction_text": batch_instruction_text,  # Add the instruction text to the batch
        "padding_mask": padding_mask,  # Add padding mask to the batch
    }
