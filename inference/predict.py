import torch
from PIL import Image
from inference.preprocess import preprocess_image
from torchvision.models import efficientnet_b4
import torch.nn as nn

def create_model():
    model = efficientnet_b4(pretrained=False)
    model.classifier = nn.Sequential(
        nn.Dropout(0.1),
        nn.Linear(1792, 256),
        nn.GELU(),
        nn.Dropout(0.1),
        nn.Linear(256, 6)
    )
    return model


def load_model(model_path: str):
    """Загружает обученную модель """

    model = create_model()
    checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint)
    model.eval()
    return model

def predict_image(model, image: Image.Image, class_names: list[str]):
    """
    Делает предсказание и возвращает [класс и вероятности]
    """
    input_tensor = preprocess_image(image)
    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
        pred_index = torch.argmax(probabilities).item()
    return class_names[pred_index], probabilities.tolist()