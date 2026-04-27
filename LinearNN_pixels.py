import numpy as np
import pandas as pd
import sklearn.metrics
import sklearn.model_selection

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.amp import GradScaler

RANDOM_SEED = 42
torch.manual_seed(RANDOM_SEED)

AGE_GROUPS = ['Young (<30)', 'Adult (30-60)', 'Senior (60+)']


class LinearNN(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )

    def forward(self, x):
        return self.net(x)


def load_img_data(file_path):
    with np.load(file_path) as data:
        imgs = data['images']
    return imgs


def load_data():
    meta_df = pd.read_csv('x_train.csv')
    y_df = pd.read_csv('y_train.csv')

    age_groups = pd.cut(meta_df['age'], bins=[0, 30, 60, 100], labels=AGE_GROUPS)

    x_dev_imgs = load_img_data('x_train_img.npz').astype(np.float32) / 255.0
    x_test_imgs = load_img_data('x_test_img.npz').astype(np.float32) / 255.0

    # Per-channel normalization fit on dev set only
    means = x_dev_imgs.mean(axis=(0, 1, 2))
    stds = x_dev_imgs.std(axis=(0, 1, 2))
    x_dev_imgs = (x_dev_imgs - means) / stds
    x_test_imgs = (x_test_imgs - means) / stds

    # Flatten HxWxC -> (N, H*W*C)
    x_dev = x_dev_imgs.reshape(len(x_dev_imgs), -1)
    x_test = x_test_imgs.reshape(len(x_test_imgs), -1)

    y = y_df['coarse_label'].values

    return x_dev, y, x_test, age_groups


def print_age_group_aucs(y_true, probs, age_groups):
    for group in AGE_GROUPS:
        mask = np.array(age_groups == group)
        n = mask.sum()
        if n == 0:
            print(f"  {group}: N/A (no samples)")
            continue
        n_pos = y_true[mask].sum()
        n_neg = n - n_pos
        if len(np.unique(y_true[mask])) < 2:
            print(f"  {group}: N/A (only one class — {n_pos} pos, {n_neg} neg)")
            continue
        auc = sklearn.metrics.roc_auc_score(y_true[mask], probs[mask])
        print(f"  {group}: AUC={auc:.4f}  (n={n}, pos={n_pos}, neg={n_neg})")


def make_loader(x, y, batch_size, shuffle, sample_weights=None):
    tensors = [torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.long)]
    if sample_weights is not None:
        tensors.append(torch.tensor(sample_weights, dtype=torch.float32))
    return DataLoader(TensorDataset(*tensors), batch_size=batch_size, shuffle=shuffle)


def upsample_young(x_tr, y_tr, age_tr, multiplier=3):
    rng = np.random.default_rng(RANDOM_SEED)
    young_mask = age_tr == 'Young (<30)'
    if young_mask.sum() == 0:
        return x_tr, y_tr, age_tr
    young_idx = np.where(young_mask)[0]
    extra_idx = rng.choice(young_idx, size=young_mask.sum() * (multiplier - 1), replace=True)
    all_idx = np.concatenate([np.arange(len(x_tr)), extra_idx])
    return x_tr[all_idx], y_tr[all_idx], age_tr[all_idx]


def compute_sample_weights(y_tr, age_tr, senior_neg_weight=3.0):
    weights = np.ones(len(y_tr), dtype=np.float32)
    senior_neg_mask = (age_tr == 'Senior (60+)') & (y_tr == 0)
    weights[senior_neg_mask] = senior_neg_weight
    return weights


def train_fold(model, trainloader, valloader, y_val_arr, num_epochs, lr, weight_decay, device):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_func = nn.CrossEntropyLoss(reduction='none')
    val_loss_func = nn.CrossEntropyLoss()
    use_amp = device.type == 'cuda'
    scaler = GradScaler('cuda', enabled=use_amp)

    best_auc = 0.0
    best_probs = None
    best_state_dict = None
    patience = 0
    stopping_epoch = num_epochs

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        for batch in trainloader:
            x_batch, y_batch = batch[0].to(device), batch[1].to(device)
            w_batch = batch[2].to(device) if len(batch) == 3 else None
            optimizer.zero_grad()
            with torch.amp.autocast('cuda', enabled=use_amp):
                per_sample_loss = loss_func(model(x_batch), y_batch)
                loss = (per_sample_loss * w_batch).mean() if w_batch is not None else per_sample_loss.mean()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            epoch_loss += loss.item()

        model.eval()
        with torch.no_grad():
            for x_val, y_val in valloader:
                logits = model(x_val.to(device))
                val_loss = val_loss_func(logits, y_val.to(device)).item()
                probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()

        auc = sklearn.metrics.roc_auc_score(y_val_arr, probs)

        if auc > best_auc:
            best_auc = auc
            best_probs = probs.copy()
            best_state_dict = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience = 0
        else:
            patience += 1

        if patience >= 40:
            stopping_epoch = epoch
            print(f"  Early stop at epoch {epoch}  best_AUC={best_auc:.4f}")
            break

        if epoch == 0 or (epoch + 1) % 20 == 0:
            print(f"  Epoch [{epoch+1}/{num_epochs}]  loss={epoch_loss/len(trainloader):.4f}  val_loss={val_loss:.4f}  AUC={auc:.4f}")

    return best_probs, stopping_epoch, best_state_dict


def run_cv(x_dev, y_dev_arr, age_groups_arr, kfold, lr, senior_neg_weight, device, save_models=False):
    oof_probs = np.zeros(len(y_dev_arr))
    fold_aucs = []
    fold_models = []

    for _, (train_idx, val_idx) in enumerate(kfold.split(x_dev, y_dev_arr)):
        x_tr, x_val = x_dev[train_idx], x_dev[val_idx]
        y_tr, y_val = y_dev_arr[train_idx], y_dev_arr[val_idx]
        age_tr = age_groups_arr[train_idx]

        x_tr, y_tr, age_tr = upsample_young(x_tr, y_tr, age_tr)
        sample_weights = compute_sample_weights(y_tr, age_tr, senior_neg_weight)

        batch_size = 256 if device.type == 'cuda' else 64
        model = LinearNN(input_dim=x_tr.shape[1]).to(device)
        best_probs, _, state_dict = train_fold(
            model,
            make_loader(x_tr, y_tr, batch_size=batch_size, shuffle=True, sample_weights=sample_weights),
            make_loader(x_val, y_val, batch_size=len(y_val), shuffle=False),
            y_val, num_epochs=300, lr=lr, weight_decay=1e-4,
            device=device
        )
        oof_probs[val_idx] = best_probs
        fold_aucs.append(sklearn.metrics.roc_auc_score(y_val, best_probs))

        if save_models:
            fold_models.append(state_dict)

    return oof_probs, fold_aucs, fold_models


def ensemble_predict(fold_models, x_test, input_dim, device):
    probs_list = []
    for state_dict in fold_models:
        model = LinearNN(input_dim=input_dim).to(device)
        model.load_state_dict(state_dict)
        model.eval()
        with torch.no_grad():
            logits = model(torch.tensor(x_test, dtype=torch.float32).to(device))
            probs_list.append(torch.softmax(logits, dim=1)[:, 1].cpu().numpy())
    return np.mean(probs_list, axis=0)


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    x_dev, y_dev, x_test, age_groups = load_data()
    y_dev_arr = np.array(y_dev)
    age_groups_arr = np.array(age_groups)
    print(f"Input dim: {x_dev.shape[1]} (flattened pixels)")

    lr               = 1e-4
    senior_neg_weight = 5.0
    print(f"Training with lr={lr:.0e}  senior_neg_weight={senior_neg_weight:.1f}")

    kfold = sklearn.model_selection.StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    oof_probs, fold_aucs, fold_models = run_cv(
        x_dev, y_dev_arr, age_groups_arr, kfold,
        lr, senior_neg_weight, device, save_models=True
    )

    oof_auc = sklearn.metrics.roc_auc_score(y_dev_arr, oof_probs)
    print(f"\nOOF AUC={oof_auc:.4f}  (fold AUCs: {[f'{a:.4f}' for a in fold_aucs]})")
    print("Age group OOF AUCs:")
    print_age_group_aucs(y_dev_arr, oof_probs, age_groups_arr)

    print("\nEnsembling 5 fold models for test predictions...")
    test_probs = ensemble_predict(fold_models, x_test, x_dev.shape[1], device)

    np.savetxt('yproba1_linearnn_pixels.txt', test_probs)
    print("Saved predictions to yproba1_linearnn_pixels.txt")


if __name__ == "__main__":
    main()
