from transformers import BertForMaskedLM, BertTokenizer
import torch

from tacit_learn.preprocessor import Preprocessor


example_input = """
C0:      16413 [1] pc=[00000000800007b0] W[r10=000000000000001b][1] R[r 2=0000000080030bf0] R[r 0=0000000000000000] inst=[00013503] ld      a0, 0(sp)
C0:      16414 [1] pc=[00000000800007b4] W[r 2=0000000080030c30][1] R[r 2=0000000080030bf0] R[r 0=0000000000000000] inst=[04010113] addi    sp, sp, 64
C0:      16415 [1] pc=[00000000800007b8] W[r 0=00000000800007bc][1] R[r 1=00000000800006d4] R[r 0=0000000000000000] inst=[00008067] ret
C0:      16419 [1] pc=[00000000800006d4] W[r10=000000000000001b][1] R[r 8=000000000000001b] R[r 0=0000000000000000] inst=[00040513] mv      a0, s0
C0:      16454 [1] pc=[00000000800006d8] W[r 1=0000000080000d5c][1] R[r 2=0000000080030c30] R[r 0=0000000000000000] inst=[00813083] ld      ra, 8(sp)
C0:      16455 [1] pc=[00000000800006dc] W[r 8=0000000080009db0][1] R[r 2=0000000080030c30] R[r 0=0000000000000000] inst=[00013403] ld      s0, 0(sp)
C0:      16456 [1] pc=[00000000800006e0] W[r 2=0000000080030c40][1] R[r 2=0000000080030c30] R[r 0=0000000000000000] inst=[01010113] addi    sp, sp, 16
C0:      16457 [1] pc=[00000000800006e4] W[r 0=00000000800006e8][1] R[r 1=0000000080000d5c] R[r 0=0000000000000000] inst=[00008067] ret
C0:      16736 [1] pc=[00000000800007d0] W[r 1=00000000800007d4][1] R[r 0=0000000000000000] R[r 0=0000000000000000] inst=[6d0000ef] jal     pc + 0x6d0
"""


# Load the tokenizer from the saved files (this will load all the configurations and "weights")
tokenizer = BertTokenizer(vocab_file="vocab.txt", model_max_length=512, do_lower_case=False)

preprocessor = Preprocessor()

encoded_input = preprocessor.encode(example_input)

# preprocessor.print_encoded(encoded_input)
print(encoded_input)

tokens = tokenizer(encoded_input, return_tensors="pt")

print(tokens)


for i in range(len(tokens["input_ids"][0])):
    token = tokenizer.decode(tokens["input_ids"][0][i])
    print(token, end=" ")
