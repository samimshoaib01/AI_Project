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

# Custom Dataset Class
class TrafficSignDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = sorted(os.listdir(root_dir))
        self.data = []

        for label, class_name in enumerate(self.classes):
            class_path = os.path.join(root_dir, class_name)
            for img_name in os.listdir(class_path):
                img_path = os.path.join(class_path, img_name)
                self.data.append((img_path, label))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, label = self.data[idx]
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, IMAGE_SIZE)

        if self.transform:
            img = self.transform(img)

        return img, label
