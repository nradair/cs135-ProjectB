import numpy as np
import pandas as pd

import sklearn.metrics
import sklearn.model_selection

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
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
np.random.seed(RANDOM_SEED)
torch.cuda.manual_seed_all(RANDOM_SEED)

class MyDataset(Dataset):
    def __init__(self, x, y=None, mean=None, std=None, augment=False):
        self.x = x
        self.y = y
        self.augment = augment

        self.transform = T.Compose([
            T.ToTensor(),  # handles HWC → CHW + scaling if uint8
        ])

        self.aug_transform = T.Compose([
            T.RandomApply([
                T.RandomPerspective(distortion_scale=0.2),
                T.RandomHorizontalFlip(),
                T.RandomVerticalFlip(),
            ], p=0.5),

            T.RandomApply([
                T.ColorJitter(brightness=0.5),
            ], p=0.5),
        ])

        self.normalize = T.Normalize(mean=mean, std=std)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        img = self.x[idx]
        if self.y is None:
            label = -1
        else:
            label = self.y[idx]

        img = self.transform(img)
        if self.augment:
            img = self.aug_transform(img)
        img = self.normalize(img)

        return img, torch.tensor(label, dtype=torch.long)
    

class CNN(nn.Module):
    def __init__(self, pretrained=False):
        super().__init__()
        if pretrained:
            # self.backbone = models.mobilenet_v3_small(weights='DEFAULT') # Last only
            # self.backbone = models.densenet121(weights='DEFAULT') # Last only best
            # self.backbone = models.efficientnet_b2(weights='DEFAULT') # Full train
            # self.backbone = models.googlenet(weights='DEFAULT') # Full @ epoch 8
            self.backbone = models.regnet_x_1_6gf(weights='DEFAULT') # Full - Best result
            
            # mobilenet, densenet, efficientnet
            # self.backbone.classifier = nn.Identity()

            # googlenet, regnet
            self.backbone.fc = nn.Identity()

            # for param in self.backbone.parameters():
            #     param.requires_grad = False
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


def train(x_train, y_train, x_val, y_val, age_groups, mean, std, model, num_train_epochs=100, batch_size=64, lr=1e-4, weight_decay=1e-4, patience_limit=30):
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

    trainloader = DataLoader(MyDataset(x_train, y_train, mean, std, augment=True),
                             batch_size=batch_size,
                             shuffle=False,
                             sampler=sampler,
                             num_workers=4,
                             pin_memory=True)
    
    validationloader = DataLoader(MyDataset(x_val, y_val, mean, std, augment=False),
                                  batch_size=batch_size,
                                  shuffle=False,
                                  num_workers=4,
                                  pin_memory=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    scheduler = ReduceLROnPlateau(optimizer, patience=1)
    
    loss_func = torch.nn.CrossEntropyLoss()

    history = {"loss": [],
              "val_loss": [],
              "auc": []}

    best_val_loss = float('inf')
    best_auc = 0
    patience = 0

    for epoch in range(num_train_epochs):
        model.train()
        epoch_loss = 0

        # if epoch == 6:
        #     for param in model.backbone.parameters():
        #         param.requires_grad = True

        for data in trainloader:
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            logits = model(inputs)
            loss = loss_func(logits, labels)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * inputs.size(0)

        model.eval()
        val_loss_total = 0
        all_labels = []
        all_probs = []

        with torch.no_grad():
            for inputs, labels in validationloader:
                inputs = inputs.to(device)
                labels = labels.to(device)

                logits = model(inputs)
                loss = loss_func(logits, labels)

                # accumulate weighted loss
                val_loss_total += loss.item() * inputs.size(0)

                probs = torch.softmax(logits, dim=1)

                all_labels.append(labels.cpu())
                all_probs.append(probs[:, 1].cpu())

        # concatenate all batches
        all_labels = torch.cat(all_labels).numpy()
        all_probs = torch.cat(all_probs).numpy()

        val_loss = val_loss_total / len(validationloader.dataset)
        auc = sklearn.metrics.roc_auc_score(all_labels, all_probs)

        scheduler.step(val_loss)

        history['loss'].append(epoch_loss / len(trainloader.dataset))
        history['val_loss'].append(val_loss)
        history['auc'].append(auc)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_auc = auc
            torch.save(model.state_dict(), "cnn.pt")
            patience = 0
        else:
            patience += 1

        if patience >= patience_limit:
            print(f"Best loss {best_val_loss} with auc {best_auc}")
            return history

        # if epoch == 0:
        #     print(f"Epoch [1/{num_train_epochs}], Train Loss: {history['loss'][epoch]:.4f}, Val Loss: {history['val_loss'][epoch]:.4f}, AUC: {history['auc'][epoch]:.4f}")
        if (epoch + 1) % 1 == 0:
            print(f"Epoch [{epoch + 1}/{num_train_epochs}], Train Loss: {history['loss'][epoch]:.4f}, Val Loss: {history['val_loss'][epoch]:.4f}, AUC: {history['auc'][epoch]:.4f}, Learning Rate: {scheduler.get_last_lr()[0]}")

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


def load_data(pretrained=False):
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

    if pretrained:
        # ImageNet stats
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
    else:
        # Dataset stats
        x_dev_temp = x_dev.copy() / 255.
        mean, std = compute_normalization_stats(x_dev_temp)

    y_labels = y_dev_df['coarse_label']

    return x_dev, y_labels, x_test, age_groups, mean, std


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


def predict(x_test, mean, std, model, batch_size=64):
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    state_dict = torch.load("cnn.pt", map_location=device)
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()

    test_dataset = MyDataset(x_test, y=None, mean=mean, std=std, augment=False)
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    all_probs = []
    with torch.no_grad():
        for batch in test_loader:
            if isinstance(batch, (list, tuple)):
                inputs = batch[0]
            else:
                inputs = batch

            inputs = inputs.to(device)
            logits = model(inputs)
            probs = torch.softmax(logits, dim=1)[:, 1]
            all_probs.append(probs.cpu())

    all_probs = torch.cat(all_probs).numpy()
    np.savetxt('yproba_cnn.txt', all_probs)


def main():
    pretrained = True
    batch_size = 64
    x_dev, y_dev, x_test, age_groups, mean, std = load_data(pretrained=pretrained)
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
    model = CNN(pretrained=pretrained)
    history = train(x_train,
                    y_train,
                    x_val,
                    y_val,
                    age_train,
                    mean,
                    std,
                    model,
                    num_train_epochs=500,
                    batch_size=batch_size,
                    lr=1e-4,
                    weight_decay=1e-4,
                    patience_limit=5)
    visualize(history)
    predict(x_test, mean, std, model, batch_size=batch_size)
    

if __name__ == "__main__":
    main()