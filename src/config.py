# config.py

# ----------------------
# Global
# ----------------------
SEED = 42

# ----------------------
# Stage-1 (Binary)
# ----------------------
STAGE1_IMG_SIZE = 224
STAGE1_BATCH_SIZE = 16
STAGE1_EPOCHS = 10
STAGE1_LR = 1e-4
STAGE1_THRESHOLD = 0.75

STAGE1_TRAIN_DIR = "data/stage-1/train"
STAGE1_VAL_DIR   = "data/stage-1/val"

STAGE1_MODEL_PATH = "checkpoints/stage1_efficientnet_b3.pth"

# ----------------------
# Stage-2 (Multi-class + Severity)
# ----------------------
STAGE2_IMG_SIZE = 224
STAGE2_BATCH_SIZE = 4  # adjust based on your GPU/CPU
STAGE2_EPOCHS = 10
STAGE2_LR = 1e-4

STAGE2_TRAIN_DIR = "data/stage-2/train"
STAGE2_VAL_DIR = "data/stage-2/val"
STAGE2_MODEL_PATH = "checkpoints/stage2_swin_cbam.pth"

NUM_STAGE2_CLASSES = 3  # DR, Cataract, Conjunctivitis
STAGE2_CLASS_NAMES = ["Diabetic_Retinopathy", "Cataract", "Conjunctivitis"]

