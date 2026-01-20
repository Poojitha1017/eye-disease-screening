import torch
from timm import create_model
from src.config import *

def get_stage2_model(num_classes):
    """
    Returns Swin Transformer model for Stage-2 classification
    """

    model = create_model(
        "swin_base_patch4_window7_224",
        pretrained=False,
        num_classes=num_classes
    )

    return model

