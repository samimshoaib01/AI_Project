import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset

# Hyperparameters
BATCH_SIZE = 32
EPOCHS = 10
IMAGE_SIZE = (32, 32)
LEARNING_RATE = 0.001
DATASET_PATH = "Dataset"
LABEL_FILE = "labels.csv"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
