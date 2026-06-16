import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.cluster import KMeans
import numpy as np
import time
from sklearn.metrics import (f1_score)

np.random.seed(42)
torch.manual_seed(42)


class RBFLayer(nn.Module):
    """
    لایه توابع پایه شعاعی (RBF)
    φ(x) = exp(-||x-c||² / (2σ²))
    """

    def __init__(self, num_inputs, num_centers, centers=None, sigma=1.0,
                 trainable_centers=False, trainable_sigma=False):
        super().__init__()
        self.num_centers = num_centers
        self.num_inputs = num_inputs

        if centers is None:
            centers = np.random.randn(num_centers, num_inputs) * 0.1

        self.centers = nn.Parameter(
            torch.tensor(centers, dtype=torch.float32),
            requires_grad=trainable_centers
        )

        if isinstance(sigma, (float, int)):
            sigma = np.full((num_centers,), sigma)

        self.sigma = nn.Parameter(
            torch.tensor(sigma, dtype=torch.float32),
            requires_grad=trainable_sigma
        )

    def forward(self, x):
        diff = x.unsqueeze(1) - self.centers.unsqueeze(0)
        distances_sq = torch.sum(diff ** 2, dim=2)
        out = torch.exp(-distances_sq / (2 * self.sigma.unsqueeze(0) ** 2))
        return out


class RBFNetwork(nn.Module):
    """
    شبکه RBF با:
    - لایه ورودی
    - لایه پنهان RBF
    - لایه خروجی خطی (برای logits)
    """

    def __init__(self, num_inputs, num_centers, num_classes=3,
                 centers=None, sigma=1.0,
                 trainable_centers=False, trainable_sigma=False):
        super().__init__()
        self.rbf = RBFLayer(num_inputs, num_centers, centers, sigma,
                            trainable_centers, trainable_sigma)
        self.output_layer = nn.Linear(num_centers, num_classes)

    def forward(self, x):
        h = self.rbf(x)
        logits = self.output_layer(h)
        return logits


def train_rbf(model, X_train, y_train, X_val, y_val, epochs=150, lr=0.01,
              batch_size=32, patience=15, verbose=True):
    """آموزش مدل RBF """
    start_time = time.time()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # تبدیل به tensor
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.long)
    X_val_t = torch.tensor(X_val, dtype=torch.float32)
    y_val_t = torch.tensor(y_val, dtype=torch.long)

    # DataLoader برای Mini-Batch
    train_dataset = TensorDataset(X_train_t, y_train_t)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    history = {'train_loss': [], 'val_loss': [], 'val_f1': []}
    best_val_f1 = 0
    best_state = None
    epochs_no_improve = 0

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        avg_train_loss = epoch_loss / len(train_loader)

        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val_t)
            val_loss = criterion(val_outputs, y_val_t)
            val_pred = torch.argmax(val_outputs, dim=1).numpy()
            val_f1 = f1_score(y_val, val_pred, average='macro')

        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(val_loss.item())
        history['val_f1'].append(val_f1)

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if verbose and (epoch + 1) % 20 == 0:
            print(
                f"Epoch {epoch + 1}/{epochs} | Loss: {avg_train_loss:.4f} | Val Loss: {val_loss.item():.4f} | Val F1: {val_f1:.4f}")

        if epochs_no_improve >= patience:
            if verbose:
                print(f" Early stopping at epoch {epoch + 1}")
            break

    model.load_state_dict(best_state)
    training_time = time.time() - start_time
    return history, best_val_f1, training_time


def initialize_centers_kmeans(X_train, num_centers):
    """مقداردهی مراکز با الگوریتم K-Means"""
    kmeans = KMeans(n_clusters=num_centers, random_state=42, n_init=10)
    kmeans.fit(X_train)
    return kmeans.cluster_centers_


def initialize_centers_random(X_train, num_centers):
    """مقداردهی تصادفی مراکز از داده‌های آموزش """
    indices = np.random.choice(len(X_train), num_centers, replace=False)
    return X_train[indices]


def get_centers(method, n, X):
    if method == 'k-means':
        return initialize_centers_kmeans(X, n)
    elif method == 'random_from_data':
        return initialize_centers_random(X, n)
    else:
        return np.random.randn(n, X.shape[1]) * 0.1
