"""
This script preprocesses a rocket RTL printf trace to reformat it into a CSV file.

example input pattern:
C0:         19 [1] pc=[0000000000010000] W[r10=0000000000010000][1] R[r 0=0000000000000000] R[r 0=0000000000000000] inst=[00000517] auipc   a0, 0x0
C0:         20 [1] pc=[0000000000010004] W[r10=0000000000010040][1] R[r10=0000000000010000] R[r 0=0000000000000000] inst=[04050513] addi    a0, a0, 64
...

example output pattern:
delta time, opcode, func3, func7, rd, rs1, rs2, imm
"""

import re
import argparse

class RiscvInstruction:
    
    LEN = 32

    def __init__(self, time, raw_inst):
        self.time = time
        self.raw_inst = raw_inst
    
    def disassemble(self):
        # get the opcode, which is the raw_inst[6:0]
        opcode = self.raw_inst[-7:]
        print(opcode)

# group time and raw inst, leave the rest 
def process_line(line):
    # extract the delta time
    parts = line.split("C0:")[1].strip()
    time = int(parts.split("[")[0].strip())

    # extract the raw inst
    # Get the hex instruction and convert to int
    inst_hex = line.split("inst=[")[1].split("]")[0].strip()
    # Convert to int, then to binary, and zero-pad to 32 bits
    raw_inst = format(int(inst_hex, 16), '032b')

    inst = RiscvInstruction(time, raw_inst)
    inst.crack()
    
    return time, raw_inst

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=str, help="input file")
    args = parser.parse_args()

    with open(args.input, "r") as f:
        lines = f.readlines()
    
    for line in lines:
        if line.startswith("C0:"):
            time, raw_inst = process_line(line)

if __name__ == "__main__":
    main()
