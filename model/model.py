import torch.nn as nn
from torchvision.models import efficientnet_b4


def create_model():
    model = efficientnet_b4(pretrained=True)

    model.classifier = nn.Sequential(
        nn.Dropout(0.1),
        nn.Linear(1792, 256),
        nn.GELU(),
        nn.Dropout(0.1),
        nn.Linear(256, 6)  # num_classes=6
    )

    return model