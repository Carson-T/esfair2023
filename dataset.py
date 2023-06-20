from torch.utils.data import Dataset
import pandas as pd
import cv2


class MyDataset(Dataset):
    def __init__(self, csv_path, transform, is_external=False):
        super(MyDataset, self).__init__()
        self.is_external = is_external
        self.csv_path = csv_path
        self.class_dict = {'BCC': 0, 'BKL': 1, 'MEL': 2, 'NV': 3, 'unknown': 4, 'VASC': 5}  # label dictionary
        self.group_dict = {"G6": 0, "G7": 1, "G8": 2, "G10": 3}
        self.transform = transform
        self.img_paths, self.labels, self.groups = self._make_dataset()  # make dataset

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        label = self.labels[idx]
        if self.is_external is False:
            group = self.groups[idx]
        else:
            group = -1
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if self.transform:
            img = self.transform(image=img)["image"]
        return img, label, group

    def _make_dataset(self):
        data = pd.read_csv(self.csv_path)
        img_paths = data["path"].values.tolist()
        labels = [self.class_dict[i] for i in data["label"].values]
        if self.is_external is False:
            groups = [self.group_dict[i] for i in data["group"].values]
        else:
            groups = None

        return img_paths, labels, groups
