import timm
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import pandas as pd
import cv2
import albumentations
from torch.utils.data import DataLoader
from albumentations import pytorch as AT
from tqdm import tqdm
from model import *
import collections

# dataset
class MyDataset(Dataset):
    def __init__(self, csv_path, transform):
        super(MyDataset, self).__init__()
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
        group = self.groups[idx]
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if self.transform:
            img = self.transform(image=img)["image"]
        return img, label, group

    def _make_dataset(self):
        data = pd.read_csv(self.csv_path)
        img_paths = data["path"].values.tolist()
        labels = [self.class_dict[i] for i in data["label"].values]
        groups = [self.group_dict[i] for i in data["group"].values]

        return img_paths, labels, groups


def test(val_loader, model, device):
    correct_nums_per_g = torch.zeros(4)
    total_nums_per_g = torch.zeros(4)
    val_loss = 0.0

    with torch.no_grad():
        for i, (images, targets, groups) in enumerate(tqdm(val_loader)):
            groups = groups.to(device)
            images = images.to(device)
            targets = targets.to(device)
            output = model(images)
            _, preds = torch.max(output, dim=1)
            if i == 0:
                all_preds = preds
                all_targets = targets
                all_groups = groups
            else:
                all_preds = torch.cat((all_preds, preds))
                all_targets = torch.cat((all_targets, targets))
                all_groups = torch.cat((all_groups, groups))
        correct = (all_preds == all_targets)
        for i in range(4):
            correct_per_g = torch.index_select(correct, 0, torch.nonzero(all_groups == i).squeeze())
            total_nums_per_g[i] = correct_per_g.shape[0]
            correct_nums_per_g[i] = torch.sum(correct_per_g).item()

    groups_acc = torch.div(correct_nums_per_g, total_nums_per_g)   # 4 groups acc
    overall_acc = torch.sum(correct_nums_per_g) / torch.sum(total_nums_per_g)  # calculate the overall accuracy

    abs_difference = 0
    minority_acc = groups_acc[3]  # minority group: G10
    for group_acc in groups_acc:
        abs_difference += abs(group_acc - minority_acc)
    SPD = abs_difference / len(groups_acc)
    fairness_score = ((0.2 - SPD) / 0.2)  # calculate the fairness score
    return val_loss / torch.sum(total_nums_per_g), groups_acc, overall_acc, fairness_score


if __name__ == '__main__':
    device = "cuda" if torch.cuda.is_available() else "cpu"

    val_transform = albumentations.Compose([
        albumentations.Resize(72, 72),
        albumentations.Normalize(),
        AT.ToTensorV2()
    ])

    val_loader = DataLoader(MyDataset("../preprocessed_data/fold1_val.csv", val_transform), batch_size=64,
                            shuffle=True, num_workers=4, pin_memory=True, drop_last=False)

    pretrained_model = timm.create_model("mobilenetv3_large_100.ra_in1k")  # convnextv2_nano.fcmae_ft_in1k  mobilenetv3_large_100.ra_in1k
    # model = myconvnext(pretrained_model, 6)
    model = MoGhoNet(pretrained_model, 6)

    # model = model.half()
    model.to(device)
    model.eval()

    #load model
    state_dict = torch.load("../saved_model/mobilenet/mobilenetv3_l-fp16-server-stu-v2.pth", map_location=device)
    new_state_dict = collections.OrderedDict()
    for name, params in state_dict.items():
        if "module" in name:
            name = name[7:]
            new_state_dict[name] = params
        else:
            new_state_dict = state_dict
            break
    del state_dict
    model.load_state_dict(new_state_dict)


    _, groups_acc, overall_acc, fairness_score = test(val_loader, model, device)
    print(overall_acc, groups_acc, fairness_score)
