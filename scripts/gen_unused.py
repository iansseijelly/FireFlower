"""
Given a vocab file, generate the rest of unused tokens to make it a total of 30522 lines
"""

import sys

vocab_file = sys.argv[1]

# First read the file to count lines
with open(vocab_file, "r") as f:
    vocab = f.readlines()
    num_lines = len(vocab)

# Calculate how many unused tokens to add
unused_lines = 30522 - num_lines
print(f"Current vocabulary has {num_lines} tokens. Adding {unused_lines} unused tokens.")

# Then append the unused tokens
with open(vocab_file, "a") as f:
    for i in range(unused_lines):
        f.write(f"[unused{1000 + i}]\n")

print(f"Updated vocabulary file now has 30522 tokens.")
