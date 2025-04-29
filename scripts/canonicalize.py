# example input:
'''
C0:         19 [1] pc=[0000000000010000] W[r10=0000000000010000][1] R[r 0=0000000000000000] R[r 0=0000000000000000] inst=[00000517] auipc   a0, 0x0
C0:         20 [1] pc=[0000000000010004] W[r10=0000000000010040][1] R[r10=0000000000010000] R[r 0=0000000000000000] inst=[04050513] addi    a0, a0, 64
'''
# example output:
'''
START INST auipc RD x11 IMM 0 TIMESTAMP 0 END
START INST addi RD x11 RS1 x11 IMM -636 TIMESTAMP 1 END
'''

# constants
BRANCH_OPCODES = ["beq", "bge", "bgeu", "blt", "bltu", "bne", "beqz", "bnez",
                    "bgez", "blez", "bltz", "bgtz", "bgt", "ble", "bgtu", "bleu",
                    "c.beqz", "c.bnez", "c.bltz", "c.bgez"]
IJ_OPCODES = ["jal", "j", "call", "tail", "c.j", "c.jal"]
UJ_OPCODES = ["jalr", "jr", "c.jr", "c.jalr", "ret"]

import re
import sys
import subprocess
import os
import glob
import tempfile
import shutil

import constants

# capture only 3 groups C0: [cycle] pc=[...] and ... inst=[...]
pattern = re.compile(r"C0:\s+(\d+)\s+.*pc=\[([0-9a-fA-F]+)\].*inst=\[([0-9a-fA-F]+)\]")

def generate_dasm_placeholders(input_file, temp_file):
    """First pass: Generate file with DASM placeholders"""
    with open(input_file, "r") as f:
        lines = f.readlines()
    
    prev_timestamp = 18  # magic number
    
    with open(temp_file, "w") as out:
        for line in lines:
            match = pattern.match(line)
            if match:
                timestamp = match.group(1)
                pc = match.group(2)[-8:]
                inst = match.group(3)

                curr_timestamp = int(timestamp) - prev_timestamp
                prev_timestamp = int(timestamp)

                # Create placeholder line
                out.write(f"START PC {pc} INST DASM(0x{inst}) TIMESTAMP {curr_timestamp} END\n")

def process_canonicalized_file(input_path, output_path, max_bb_size):
    """Third pass: Process the canonicalized file and add basic blocks"""
    with open(input_path, "r") as f:
        lines = f.readlines()
    
    current_bb = []
    bb_timestamp = 0
    
    with open(output_path, "w") as out:
        for line in lines:
            if not line.startswith("START"):
                continue
            
            # Parse the instruction
            parts = line.strip().split()
            if len(parts) < 4:
                continue  # Skip malformed lines
                
            # Find INST index and extract opcode
            try:
                inst_idx = parts.index("INST")
                opcode = parts[inst_idx + 1] if inst_idx + 1 < len(parts) else ""
                
                # Find TIMESTAMP index and extract value
                timestamp_idx = parts.index("TIMESTAMP")
                timestamp = int(parts[timestamp_idx + 1]) if timestamp_idx + 1 < len(parts) else 0
            except ValueError:
                continue  # Skip lines missing required components
            
            # Add this instruction to the current basic block
            current_bb.append(line)
            bb_timestamp += timestamp
            
            # Check if the opcode is a branch, jump, or return opcode
            if opcode in BRANCH_OPCODES or opcode in IJ_OPCODES or opcode in UJ_OPCODES:
                # Only output basic block if it's within size threshold
                if len(current_bb) <= max_bb_size:
                    for inst in current_bb:
                        out.write(inst)
                    out.write(f"BBTIME {bb_timestamp} ENDBB\n")
                
                # Reset for the next basic block
                current_bb = []
                bb_timestamp = 0
        
        # Handle the last basic block if there's anything left
        if current_bb and len(current_bb) <= max_bb_size:
            for inst in current_bb:
                out.write(inst)
            out.write("BBTIME 0 ENDBB\n")  # No timestamp for final BB

if __name__ == "__main__":



    # Set max basic block size threshold
    MAX_BB_SIZE = constants.MAX_BB_SIZE  # Adjust this value as needed
    
    # Create output and temp directories
    os.makedirs("data/canonicalized", exist_ok=True)
    temp_dir = tempfile.mkdtemp()
    
    try:
        # go to pwd
        script_dir = os.path.dirname(os.path.abspath(__file__))
        os.chdir(script_dir)

        input_files = glob.glob("../data/orig/*")
        
        if not input_files:
            print("No files found in data/orig directory.")
            sys.exit(1)
        
        for input_file in input_files:
            basename = os.path.basename(input_file)
            print(f"Processing {basename}...")
            
            # First pass: Generate DASM placeholders
            temp_file = os.path.join(temp_dir, f"{basename}.tmp")
            generate_dasm_placeholders(input_file, temp_file)
            print(f"DASM placeholders generated in {temp_file}")
            
            # Second pass: Run dasm_all to convert to canonical form
            canonical_file = os.path.join(temp_dir, f"{basename}.canonical.out")
            subprocess.run(["./dasm_all", "-i", temp_file, "-c", "-o", canonical_file], 
                           check=True)
            print(f"Canonicalized file generated in {canonical_file}")
            
            # Third pass: Process canonicalized file and add basic blocks
            output_file = os.path.join("../data/canonicalized", f"{basename}.out")
            process_canonicalized_file(canonical_file, output_file, MAX_BB_SIZE)
            print(f"Basic blocks added to {output_file}")
            
        print(f"Processing complete. Output written to data/canonicalized/")
    
    finally:
        # Clean up temp directory
        shutil.rmtree(temp_dir)