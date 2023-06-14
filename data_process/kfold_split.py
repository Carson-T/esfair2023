import pandas as pd
import os
import numpy as np
from sklearn.model_selection import StratifiedKFold

classes = ['BCC', 'BKL', 'MEL', 'NV', 'unknown', 'VASC']  # label dictionary
G = ["G6", "G7", "G8", "G10"]
path = r"..\..\preprocessed_data\TrainingSet"
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=2023)

img_paths = []
labels = []
groups = []
for group in G:
    group_dir = os.path.join(path, group)
    for class_name in classes:
        class_dir = os.path.join(group_dir, class_name)
        for file_name in os.listdir(class_dir):
            if file_name.endswith('.jpg') or file_name.endswith('.png'):
                img_path = os.path.join(class_dir, file_name)[3:]
                img_paths.append(img_path)
                labels.append(class_name)
                groups.append(group)

img_paths = np.array(img_paths)
labels = np.array(labels)
groups = np.array(groups)
for k, (train_idx, val_idx) in enumerate(kf.split(img_paths, labels)):
    train_path = img_paths[train_idx]
    val_path = img_paths[val_idx]
    train_labels = labels[train_idx]
    val_labels = labels[val_idx]
    train_groups = groups[train_idx]
    val_groups = groups[val_idx]
    train_df = pd.DataFrame()
    val_df = pd.DataFrame()

    train_df.insert(loc=len(train_df.columns), column="path", value=train_path)
    train_df.insert(loc=len(train_df.columns), column="label", value=train_labels)
    train_df.insert(loc=len(train_df.columns), column="group", value=train_groups)
    train_df.to_csv(f"../../preprocessed_data/fold{k + 1}_train.csv", index=False)
    val_df.insert(loc=len(val_df.columns), column="path", value=val_path)
    val_df.insert(loc=len(val_df.columns), column="label", value=val_labels)
    val_df.insert(loc=len(val_df.columns), column="group", value=val_groups)
    val_df.to_csv(f"../../preprocessed_data/fold{k + 1}_val.csv", index=False)
