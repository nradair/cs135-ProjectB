import os
import numpy as np
import pandas as pd
import random

import sklearn.metrics
import sklearn.model_selection

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torch.nn.functional as F
import torchvision.transforms as T

from matplotlib import pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings('ignore')

plt.rcParams['figure.figsize'] = (10.0, 8.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

RANDOM_SEED = 42
torch.manual_seed(RANDOM_SEED)

class MyDataset(Dataset):
    def __init__(self, x, y, augment_negatives=False):
        self.x = torch.tensor(x, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

        self.augment_negatives = augment_negatives

        # Base transform (always applied)
        self.base_transform = T.Compose([
            T.Lambda(lambda img: img.permute(2, 0, 1)),  # HWC -> CHW
        ])

        # Augmentation for negative class (label == 0)
        self.neg_transform = T.Compose([
            T.Lambda(lambda img: img.permute(2, 0, 1)),
            T.RandomRotation(degrees=90),
        ])

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        img = self.x[idx]
        label = self.y[idx]

        if self.augment_negatives and label == 0:
            img = self.neg_transform(img)
        else:
            img = self.base_transform(img)

        return img, label
    

def create_balanced_sampler(labels):
    labels = torch.tensor(labels)

    class_counts = torch.bincount(labels)
    class_weights = 1.0 / class_counts.float()

    sample_weights = class_weights[labels]

    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(labels),
        replacement=True
    )

    return sampler


class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.seq = nn.Sequential(
            nn.LazyConv2d(16, 3, padding='same'),
            nn.LazyBatchNorm2d(),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.LazyConv2d(32, 3, padding='same'),
            nn.LazyBatchNorm2d(),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.LazyConv2d(64, 3, padding='same'),
            nn.LazyBatchNorm2d(),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.LazyConv2d(128, 3, padding='same'),
            nn.LazyBatchNorm2d(),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),

            nn.Flatten(),
            nn.LazyLinear(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.LazyLinear(2)
        )

    def forward(self, x):
        logits = self.seq(x)
        return logits


def train(x_train, y_train, x_val, y_val, model, num_train_epochs, batch_size, lr, weight_decay):
    """
    args:
      x_train: `np.array((N, D))`, training data of N instances and D features.
      y_train: `np.array((N, C))`, training labels of N instances and C fitting targets 
      x_val: `np.array((N1, D))`, validation data of N1 instances and D features.
      y_val: `np.array((N1, C))`, validation labels of N1 instances and C fitting targets 
      model: a torch module
      num_train_epochs: int, the number of training epochs.
      batch_size: int, the batch size 
      lr: float, learning rate
      weight_decay: float, weight decay for regularization 
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    model = model.to(device)

    sampler = create_balanced_sampler(y_train)

    trainloader = DataLoader(MyDataset(x_train, y_train, augment_negatives=True),
                             batch_size=batch_size,
                             shuffle=False,
                             sampler=sampler)
    
    validationloader = DataLoader(MyDataset(x_val, y_val, augment_negatives=False),
                                  batch_size=y_val.shape[0],
                                  shuffle=False)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    loss_func = torch.nn.CrossEntropyLoss()

    history = {"loss": [],
              "val_loss": [],
              "accuracy": [],
              "auc": []}

    best_val_loss = float('inf')
    best_accuracy = 0
    best_auc = 0

    # patience variables
    improved = False
    patience = 0

    for epoch in range(num_train_epochs):
        epoch_loss = 0
        for data in trainloader:
            model.train()
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            logits = model(inputs)
            loss = loss_func(logits, labels)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        with torch.no_grad():
            for data in validationloader:
                model.eval()
                inputs, labels = data
                inputs = inputs.to(device)
                labels = labels.to(device)
                logits = model(inputs)
                val_loss = loss_func(logits, labels).detach()

                probs = torch.softmax(logits, dim=1)
                accuracy = sklearn.metrics.accuracy_score(labels.numpy(), torch.argmax(probs, dim=1).numpy())
                auc = sklearn.metrics.roc_auc_score(labels.numpy(), probs[:,1].numpy())

                history['loss'].append(epoch_loss / (y_train.shape[0] / batch_size))
                history['val_loss'].append(val_loss)
                history['accuracy'].append(accuracy)
                history['auc'].append(auc)

                if val_loss < best_val_loss:
                    improved = True
                    best_val_loss = val_loss
                    best_accuracy = accuracy
                    best_auc = auc
                    torch.save(model.state_dict(), "cnn.pt")
                else:
                    improved = False

                # Early stopping
                patience += 1
                if improved:
                    patience = 0
                elif patience >= 30:
                    print(f"Best loss {best_val_loss} with accuracy {best_accuracy} and auc {best_auc}")
                    return history

        if epoch == 0:
            print(f"Epoch [1/{num_train_epochs}], Train Loss: {history['loss'][epoch]:.4f}, Val Loss: {history['val_loss'][epoch]:.4f}, Acc Score: {history['accuracy'][epoch]:.4f}, AUC: {history['auc'][epoch]:.4f}")
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch + 1}/{num_train_epochs}], Train Loss: {history['loss'][epoch]:.4f}, Val Loss: {history['val_loss'][epoch]:.4f}, Acc Score: {history['accuracy'][epoch]:.4f}, AUC: {history['auc'][epoch]:.4f}")

    return history


# Function and code to load images
def load_img_data(file_path):
    with np.load(file_path) as data:
        img = data['images']
        ids = data['image_ids']
    print(f"Successfully loaded {img.shape[0]} images.")
    return img, ids


def compute_normalization_stats(x):
    """
    Compute per-channel normalization stats for a 4D tensor of np images (0, 255)
    """
    channel_means = x.mean(axis=(0, 1, 2))
    channel_stds = x.std(axis=(0, 1, 2))

    # print("RGB Means:", channel_means)
    # print("RGB Stds:", channel_stds)

    return channel_means, channel_stds


def load_data():
    # x_dev_df = pd.read_csv('x_train.csv')
    y_dev_df = pd.read_csv('y_train.csv')
    # x_test_df = pd.read_csv('x_test.csv')

    x_dev, dev_ids = load_img_data('x_train_img.npz')
    x_test, test_ids = load_img_data('x_test_img.npz')
    x_dev = x_dev / 255.
    x_test = x_test / 255.

    (m0, m1, m2), (std0, std1, std2) = compute_normalization_stats(x_dev)
    x_dev[:,:,:,0] = (x_dev[:,:,:,0] - m0) / std0
    x_dev[:,:,:,1] = (x_dev[:,:,:,1] - m1) / std1
    x_dev[:,:,:,2] = (x_dev[:,:,:,2] - m2) / std2
    x_test[:,:,:,0] = (x_test[:,:,:,0] - m0) / std0
    x_test[:,:,:,1] = (x_test[:,:,:,1] - m1) / std1
    x_test[:,:,:,2] = (x_test[:,:,:,2] - m2) / std2

    y_labels = y_dev_df['coarse_label']

    return x_dev, y_labels, x_test


def visualize(history):
    plt.subplot(2, 1, 1)
    plt.plot(history['loss'], label='train')
    plt.plot(history['val_loss'], label='val')
    plt.legend()
    plt.xlabel('Epochs')
    plt.ylabel('Loss')

    plt.subplot(2, 1, 2)
    plt.plot(history['accuracy'], label='val accuracy')
    plt.ylim(0.0, 1.0)
    plt.xlabel('Epoch')
    plt.ylabel('Clasification accuracy')
    plt.legend()
    plt.show()


def predict(x_test):
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    state_dict = torch.load("cnn.pt")
    model = CNN()
    model.load_state_dict(state_dict)
    model.eval()
    y_probs = model(torch.tensor(x_test).permute(0, 3, 1, 2).astype(np.float32))[:, 1]
    np.savetxt('yproba_cnn.txt', y_probs.detach().numpy())


def main():
    x_dev, y_dev, x_test = load_data()
    x_train, x_val, y_train, y_val = sklearn.model_selection.train_test_split(x_dev, y_dev, test_size=0.20, random_state=RANDOM_SEED)
    model = CNN()
    history = train(x_train,
                    y_train,
                    x_val,
                    y_val,
                    model,
                    num_train_epochs=80,
                    batch_size=64,
                    lr=1e-4,
                    weight_decay=1e-3)
    visualize(history)
    predict(x_test)
    

if __name__ == "__main__":
    main()