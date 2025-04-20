CHECKPOINT_PATH = "checkpoints/model_final.pth"  # Path to your trained model
DATA_FOLDER = "data/canonicalized"  # Test/validation data file
VOCAB_FILE = "vocab/opcodes.txt"
BATCH_SIZE = 64  
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RESULTS_DIR = "./evaluation_results"
SAMPLE_BATCHES = 100  # Number of batches to evaluate on (set to None to use all data)