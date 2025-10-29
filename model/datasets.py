import os
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from config import TRAIN_DATA_PATH, train_transforms


class ImageDataset(Dataset):
    def __init__(self, data, root_path, transform=None):
        self.data = data
        self.root_path = root_path
        self.transform = transform
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image_name = self.data.iloc[idx]['filename']
        label = self.data.iloc[idx]['label']

        image_path = os.path.join(self.root_path, image_name)
        image = Image.open(image_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return {
            'image': image,
            'label': torch.tensor(label, dtype=torch.long)
        }

def create_train_dataloader(data, batch_size=32):
    train_dataset = ImageDataset(data, TRAIN_DATA_PATH, train_transforms)
    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True, pin_memory=True)
    return train_dataloader
