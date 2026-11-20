import os
import torch

class CFG:
    TRAIN_PATH = "./data/train.csv"
    TEST_PATH = "./data/test.csv"
    ARTIFACT_DIR = "./artifacts"
    SUB_PATH = "./submission.csv"

    SEED = 42
    N_FOLDS = 5

    # Giảm BATCH_SIZE xuống 256 nếu GPU có <= 4GB VRAM
    BATCH_SIZE = 512
    EPOCHS = 40
    LR = 1e-3
    WEIGHT_DECAY = 1e-4
    EARLY_STOPPING_ROUNDS = 6

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    USE_AMP = True if torch.cuda.is_available() else False

os.makedirs(CFG.ARTIFACT_DIR, exist_ok=True)