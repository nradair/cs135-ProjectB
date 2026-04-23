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
import torchvision.models as models

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
        self.y = torch.tensor(np.array(y), dtype=torch.long)

        self.augment_negatives = augment_negatives

        # Base transform (always applied)
        self.base_transform = T.Compose([
            T.Lambda(lambda img: img.permute(2, 0, 1)),  # HWC -> CHW
        ])

        # Augmentation for negative class (label == 0)
        self.neg_transform = T.RandomApply(nn.ModuleList([
            # T.RandomRotation(degrees=90),
            T.RandomPerspective(distortion_scale=0.2, p=1),
            T.RandomVerticalFlip(p=1),
            T.RandomHorizontalFlip(p=1),
            T.ColorJitter(brightness=0.5)
            # T.GaussianBlur(3)
        ]), p=0.25)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        img = self.x[idx]
        label = self.y[idx]

        img = self.base_transform(img)

        if self.augment_negatives and label == 0:
            img = self.neg_transform(img)

        return img, label
    

class CNN(nn.Module):
    def __init__(self, pretrained=False):
        super().__init__()
        if pretrained:
            self.backbone = models.mobilenet_v3_small(weights='DEFAULT')
        else:
            self.backbone = nn.Sequential(
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
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.LazyConv2d(256, 3, padding='same'),
                nn.LazyBatchNorm2d(),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.LazyConv2d(512, 3, padding='same'),
                nn.LazyBatchNorm2d(),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.LazyConv2d(1024, 3, padding='same'),
                nn.LazyBatchNorm2d(),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.LazyConv2d(2048, 3, padding='same'),
                nn.LazyBatchNorm2d(),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d(1)
            )

        self.head = nn.Sequential(
            nn.Flatten(),
            nn.LazyLinear(512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.LazyLinear(2)
        )

    def forward(self, x):
        feats = self.backbone(x)
        logits = self.head(feats)
        return logits


def create_balanced_sampler(labels, age_groups):
    labels = torch.tensor(labels)

    age_group_codes = torch.tensor(pd.Categorical(age_groups).codes)
    n_age_groups = len(torch.unique(age_group_codes))
    joint_groups = labels * n_age_groups + age_group_codes
    group_counts = torch.bincount(joint_groups)

    group_weights = 1.0 / group_counts.float()
    group_weights[group_counts == 0] = 0
    group_weights = torch.clamp(group_weights, max=10.0)

    sample_weights = group_weights[joint_groups]

    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )

    return sampler


def train(x_train, y_train, x_val, y_val, age_groups, model, num_train_epochs, batch_size, lr, weight_decay):
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

    sampler = create_balanced_sampler(y_train, age_groups)

    trainloader = DataLoader(MyDataset(x_train, y_train, augment_negatives=False),
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
              "auc": []}

    best_val_loss = float('inf')
    best_auc = 0
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
                auc = sklearn.metrics.roc_auc_score(labels.cpu().numpy(), probs[:,1].cpu().numpy())

                history['loss'].append(epoch_loss / len(trainloader))
                history['val_loss'].append(val_loss.cpu())
                history['auc'].append(auc)

                if val_loss < best_val_loss:
                    improved = True
                    best_val_loss = val_loss
                    best_auc = auc
                    torch.save(model.state_dict(), "cnn.pt")
                    patience = 0
                else:
                    patience += 1

                # Early stopping
                if patience >= 30:
                    print(f"Best loss {best_val_loss} with auc {best_auc}")
                    return history

        if epoch == 0:
            print(f"Epoch [1/{num_train_epochs}], Train Loss: {history['loss'][epoch]:.4f}, Val Loss: {history['val_loss'][epoch]:.4f}, AUC: {history['auc'][epoch]:.4f}")
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch + 1}/{num_train_epochs}], Train Loss: {history['loss'][epoch]:.4f}, Val Loss: {history['val_loss'][epoch]:.4f}, AUC: {history['auc'][epoch]:.4f}")

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
    x_dev_df = pd.read_csv('x_train.csv')
    y_dev_df = pd.read_csv('y_train.csv')
    # x_test_df = pd.read_csv('x_test.csv')

    age_groups = pd.cut(
                    x_dev_df['age'],
                    bins=[0, 30, 60, 100],
                    labels=['Young (<30)', 'Adult (30-60)', 'Senior (60+)']
                )

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

    return x_dev, y_labels, x_test, age_groups


def visualize(history):
    plt.subplot(2, 1, 1)
    plt.plot(history['loss'], label='train')
    plt.plot(history['val_loss'], label='val')
    plt.legend()
    plt.xlabel('Epochs')
    plt.ylabel('Loss')

    plt.subplot(2, 1, 2)
    plt.plot(history['auc'], label='val AUC')
    plt.ylim(0.0, 1.0)
    plt.xlabel('Epoch')
    plt.ylabel('ROC AUC')
    plt.legend()

    plt.savefig('cnn.png')
    # plt.show()


def predict(x_test, model):
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    state_dict = torch.load("cnn.pt")
    model.load_state_dict(state_dict)
    model.eval()
    model = model.to(device)
    inputs = torch.tensor(x_test, dtype=torch.float32).to(device)
    logits = model(inputs.permute(0, 3, 1, 2))
    probs = torch.softmax(logits, dim=1)[:, 1]
    np.savetxt('yproba_cnn.txt', probs.detach().cpu().numpy())


def main():
    x_dev, y_dev, x_test, age_groups = load_data()
    x_train, x_val, y_train, y_val, age_train, age_val = sklearn.model_selection.train_test_split(
        x_dev,
        y_dev,
        age_groups,
        test_size=0.20,
        stratify=y_dev,
        random_state=RANDOM_SEED
    )
    y_train = y_train.to_numpy()
    y_val = y_val.to_numpy()
    model = CNN(pretrained=False)
    history = train(x_train,
                    y_train,
                    x_val,
                    y_val,
                    age_train,
                    model,
                    num_train_epochs=500,
                    batch_size=64,
                    lr=1e-4,
                    weight_decay=1e-3)
    visualize(history)
    predict(x_test, model)
    

if __name__ == "__main__":
    main()