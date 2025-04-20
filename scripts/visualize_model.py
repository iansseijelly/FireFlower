# visualize the model using torchviz

import torch
from tacit_learn.model import FireFlowerPredictor, FireFlowerConfig
from tacit_learn.dataloader import BasicBlockDataset, collate_fn
from torch.utils.data import DataLoader
from math import sqrt
from torchviz import make_dot
DATA_FILE = "data/rocket-hello.canonicalized.out"  # Test/validation data file
VOCAB_FILE = "vocab/opcodes.txt"
BATCH_SIZE = 1
CHECKPOINT_PATH = "checkpoints/model_final.pth"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

test_dataset = BasicBlockDataset(vocab_path=VOCAB_FILE, file_path=DATA_FILE)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, collate_fn=collate_fn, shuffle=False)


# Initialize model with the same configuration as during training
NUM_INST = test_dataset.get_n_inst()
NUM_REGS = test_dataset.get_n_reg()
MAX_BLOCK_LEN = 64
config = FireFlowerConfig(n_inst=NUM_INST, 
                          n_reg=NUM_REGS,
                          d_inst=int(sqrt(NUM_INST)),
                          d_reg=int(sqrt(NUM_REGS)),
                          d_imm=int(sqrt(NUM_INST)),
                          d_bb=int(sqrt(NUM_INST)),
                          d_model=128,
                          n_head=4,
                          n_layers=2,
                          n_pos=MAX_BLOCK_LEN,
                          d_pos=int(sqrt(NUM_REGS)))

model = FireFlowerPredictor(config)
model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE))

# get a batch from the test loader
batch = next(iter(test_loader))

# visualize the model
make_dot(model(batch), params=dict(model.named_parameters()))