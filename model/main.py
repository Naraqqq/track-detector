import torch
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from model import create_model
from sklearn.metrics import f1_score
from datasets import create_train_dataloader
from sklearn.preprocessing import LabelEncoder
from config import TRAIN_DATA_PATH, BATCH_SIZE, NUM_EPOCHS, device



def train():
    labels_path = TRAIN_DATA_PATH / "_classes.csv"
    data = pd.read_csv(labels_path).copy()
    data.columns = ['filename', 'Bear', 'Bird', 'Cat', 'Dog', 'Leopard',
                    'Otter']  # Названия колонок были " Bear", " Bird", " Cat" и тд.
    data['label'] = data.drop('filename', axis=1).idxmax(axis=1)
    le = LabelEncoder()
    data['label'] = le.fit_transform(data['label'])
    data.drop(columns=['Bear', 'Bird', 'Cat', 'Dog', 'Leopard', 'Otter'], inplace=True)

    train_dataloader = create_train_dataloader(data, BATCH_SIZE)

    model = create_model()
    model.to(device)

    # Веса для несбалансированных классов
    weights = 1 / data['label'].value_counts().sort_index().values
    weights = weights / weights.min()
    weights = torch.tensor(weights, dtype=torch.float32, device=device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30)
    criterion = torch.nn.CrossEntropyLoss(weight=weights, label_smoothing=0.05)

    train_metrics = []
    val_metrics = []
    mean_losses = []

    # Основной цикл обучения
    for epoch in tqdm(range(NUM_EPOCHS)):
        model.train()
        running_loss = 0.0
        train_predictions = []
        train_targets = []
        num_batches = 0

        for batch in train_dataloader:
            optimizer.zero_grad()
            images, labels = batch['image'].to(device), batch['label'].to(device)

            predictions = model(images)
            loss = criterion(predictions, labels)
            preds = predictions.argmax(dim=1).cpu().numpy()

            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            train_predictions.extend(preds)
            train_targets.extend(labels.cpu().numpy())
            num_batches += 1

        scheduler.step()
        mean_losses.append(running_loss / num_batches)
        train_f1 = f1_score(train_targets, train_predictions, average='macro')
        train_metrics.append(train_f1)

        plt.figure(figsize=(10, 4))
        plt.plot(train_metrics, label='Train F1', color='r')
        plt.title('F1 Score')
        plt.legend()
        plt.tight_layout()
        plt.show()
        # torch.save(model.state_dict(), "./efficientnet_model.pth")


if __name__ == "__main__":
    train()