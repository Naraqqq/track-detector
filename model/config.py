import torch
from torchvision import transforms
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

NUM_EPOCHS = 30
BATCH_SIZE = 32
LEARNING_RATE = 5e-4

TRAIN_DATA_PATH = PROJECT_ROOT / "imgs" / "train"


train_transforms = transforms.Compose([
    transforms.Resize((384, 384)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.2),
    transforms.RandomRotation(degrees=20),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
    transforms.RandAugment(num_ops=3, magnitude=15),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    transforms.RandomErasing(p=0.2, scale=(0.02, 0.2))
])