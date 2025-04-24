import cv2
import torch
import numpy as np
import pandas as pd
from torchvision import transforms
import torch.nn as nn  # <-- Missing import added

# Load model
MODEL_PATH = 'traffic_sign_cnn.pth'
LABEL_FILE = 'labels.csv'
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load class names
class_df = pd.read_csv(LABEL_FILE)
class_names = class_df['Name'].tolist()

