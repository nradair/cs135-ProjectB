import numpy as np
import pandas as pd
import sklearn.metrics
import sklearn.model_selection
import sklearn.neighbors
import sklearn.preprocessing

RANDOM_SEED = 42


AGE_GROUPS = ['Young (<30)', 'Adult (30-60)', 'Senior (60+)']


def load_data():
    x_train_df = pd.read_csv('x_train_engineered_features.csv')
    x_test_df = pd.read_csv('x_test_engineered_features.csv')
    y_df = pd.read_csv('y_train.csv')
    meta_df = pd.read_csv('x_train.csv')

    age_groups = pd.cut(meta_df['age'], bins=[0, 30, 60, 100], labels=AGE_GROUPS)

    x_train = x_train_df.select_dtypes(include=[np.number]).values
    x_test = x_test_df.select_dtypes(include=[np.number]).values
    y = y_df['coarse_label'].values

    return x_train, y, x_test, age_groups


def print_age_group_aucs(y_true, probs, age_groups):
    for group in AGE_GROUPS:
        mask = np.array(age_groups == group)
        n = mask.sum()
        if n == 0:
            print(f"  {group}: N/A (no samples in val)")
            continue
        n_pos = y_true[mask].sum()
        n_neg = n - n_pos
        if len(np.unique(y_true[mask])) < 2:
            print(f"  {group}: N/A (only one class in val — {n_pos} pos, {n_neg} neg)")
            continue
        auc = sklearn.metrics.roc_auc_score(y_true[mask], probs[mask])
        print(f"  {group}: AUC={auc:.4f}  (n={n}, pos={n_pos}, neg={n_neg})")


def main():
    x_dev, y_dev, x_test, age_groups = load_data()
    y_dev_arr = np.array(y_dev)
    age_groups_arr = np.array(age_groups)

    kfold = sklearn.model_selection.StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)

    best_mean_auc = 0
    best_k = None

    for k in [3, 5, 7, 11, 15, 21, 31, 51, 71, 101]:
        oof_probs = np.zeros(len(y_dev_arr))
        fold_aucs = []

        for train_idx, val_idx in kfold.split(x_dev, y_dev_arr):
            x_tr, x_val = x_dev[train_idx], x_dev[val_idx]
            y_tr, y_val = y_dev_arr[train_idx], y_dev_arr[val_idx]

            scaler = sklearn.preprocessing.StandardScaler()
            x_tr = scaler.fit_transform(x_tr)
            x_val = scaler.transform(x_val)

            model = sklearn.neighbors.KNeighborsClassifier(n_neighbors=k, metric='euclidean', n_jobs=-1)
            model.fit(x_tr, y_tr)
            val_probs = model.predict_proba(x_val)[:, 1]
            oof_probs[val_idx] = val_probs
            fold_aucs.append(sklearn.metrics.roc_auc_score(y_val, val_probs))

        mean_auc = np.mean(fold_aucs)
        oof_auc = sklearn.metrics.roc_auc_score(y_dev_arr, oof_probs)
        print(f"k={k:3d}  CV AUC={mean_auc:.4f} ± {np.std(fold_aucs):.4f}  OOF AUC={oof_auc:.4f}")
        print_age_group_aucs(y_dev_arr, oof_probs, age_groups_arr)

        if mean_auc > best_mean_auc:
            best_mean_auc = mean_auc
            best_k = k

    print(f"\nBest k={best_k} with CV AUC={best_mean_auc:.4f}")

    scaler = sklearn.preprocessing.StandardScaler()
    x_dev_scaled = scaler.fit_transform(x_dev)
    x_test_scaled = scaler.transform(x_test)

    final_model = sklearn.neighbors.KNeighborsClassifier(n_neighbors=best_k, metric='euclidean', n_jobs=-1)
    final_model.fit(x_dev_scaled, y_dev_arr)

    test_probs = final_model.predict_proba(x_test_scaled)[:, 1]
    np.savetxt('yproba1_knn.txt', test_probs)
    print("Saved predictions to yproba1_knn.txt")


if __name__ == "__main__":
    main()
