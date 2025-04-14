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

import re
import sys
import subprocess
# capture only 3 groups C0: [cycle] pc=[...] and ... inst=[...]
pattern = re.compile(r"C0:\s+(\d+)\s+.*pc=\[([0-9a-fA-F]+)\].*inst=\[([0-9a-fA-F]+)\]")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python canonicalize.py <input_file>")
        sys.exit(1)

    input_file = sys.argv[1]

    with open(input_file, "r") as f:
        lines = f.readlines()
    
    prev_timestamp = 18 # magic number

    for line in lines:
        match = pattern.match(line)
        if match:
            timestamp = match.group(1)
            pc = match.group(2)[-8:]
            inst = match.group(3)

            curr_timestamp = int(timestamp) - prev_timestamp
            prev_timestamp = int(timestamp)

            # call ./dasm on inst
            output = subprocess.run(["./dasm_one", "--input", inst, "--canonical"], capture_output=True, text=True)
            # get the output
            canonical_inst = output.stdout.split("\n")[0]
            print(f"START PC {pc} INST {canonical_inst} TIMESTAMP {curr_timestamp} END")

