import os
import torch
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import pandas as pd

class CascadeImageDataset(Dataset):
    def __init__(self, excel_file, sheet_name, transform=None):
        self.transform = transform
        self.image_paths = []
        self.labels = []
        self._load_data_from_excel(excel_file, sheet_name)

    def _load_data_from_excel(self, excel_file, sheet_name):
        """Load all feature labels"""
        data = pd.read_excel(excel_file, sheet_name=sheet_name)
        self.image_paths = data['ImagePath'].tolist()
        # Load all label columns
        self.labels = data[['plaque', 'form', 'surface', 'echo', 'calcification', 'stenosis', 'vulnerability']].values

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        raw_labels = self.labels[idx]

        plaque_label = raw_labels[0] 
        if plaque_label == 0:
            multi_labels = [-1] * 5
            vul_label    = 0.0  # Set to 0 or other value
        else:
            multi_labels = raw_labels[1:6]
            vul_label    = raw_labels[6]
            
        if np.isnan(vul_label):  # Check if NaN to avoid passing NaN
            print(f"Warning: vul_label is NaN at index {idx}, setting to 0.0")
            vul_label = 0.0  # Or choose another value

        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)

        return image, {
            'plaque': torch.tensor(plaque_label, dtype=torch.float32),
            'multi_task': torch.tensor(multi_labels, dtype=torch.long),
            'vulnerability': torch.tensor(vul_label, dtype=torch.float32)
        }
