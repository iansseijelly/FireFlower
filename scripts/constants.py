import torch
import datetime


# =====================
#   Constants
# =====================
CHECKPOINT_PATH = "checkpoints/model_final.pth"  # Path to your trained model
DATA_FOLDER = "data/canonicalized"  # Test/validation data file
VOCAB_FILE = "vocab/opcodes.txt"
BATCH_SIZE = 256  
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RESULTS_DIR = "./evaluation_results"
SAMPLE_BATCHES = 100  # Number of batches to evaluate on (set to None to use all data)
LOG_DIR = "./logs/ff_training_" + datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

# =====================
#   Hyperparameters
# =====================
NUM_EPOCHS = 20
PATIENCE = 5
LEARNING_RATE = 1e-4

# Weight for the block-level sum constraint.
REG_LOSS_WEIGHT = 1
BB_LOSS_WEIGHT = 0.3
FP_LOSS_WEIGHT = 0.0
FN_LOSS_WEIGHT = 0.0

MAX_BLOCK_LEN = 64

# model parameters
D_MODEL = 128
N_HEAD = 8
N_LAYERS = 4
