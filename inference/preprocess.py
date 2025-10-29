from torchvision import transforms
from torchvision.transforms import v2
from PIL import Image
import torch

def preprocess_image(image: Image.Image):
    """Подготавливает изображение для подачи в модель"""
    img_transforms = v2.Compose([
        v2.ToPILImage(),
        v2.Resize((384, 384)),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])

    ])
    return img_transforms(image).unsqueeze(0)