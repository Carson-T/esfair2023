import os
from collections import OrderedDict
from tqdm import tqdm
# from torch import max, cat, index_select, nonzero, zeros, div, sum, load, no_grad, cuda
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import cv2
import albumentations
from albumentations import pytorch as AT
from timm import create_model


# dataset
class MyDataset(Dataset):
    def __init__(self, path, transform):
        super(MyDataset, self).__init__()
        self.path = path
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
        img_paths = []
        labels = []
        groups = []
        for group_name in self.group_dict:
            group_dir = os.path.join(self.path, group_name)
            group = self.group_dict[group_name]
            for class_name in self.class_dict:
                class_dir = os.path.join(group_dir, class_name)
                label = self.class_dict[class_name]
                for file_name in os.listdir(class_dir):
                    if file_name.endswith('.jpg') or file_name.endswith('.png'):
                        img_path = os.path.join(class_dir, file_name)
                        img_paths.append(img_path)
                        labels.append(label)
                        groups.append(group)
        return img_paths, labels, groups


# model define
class MyNet(nn.Module):
    def __init__(self, pretrained_model, num_classes):
        super(MyNet, self).__init__()
        self.pretrained_model = pretrained_model
        self.classifier = nn.Sequential(
            nn.Linear(self.pretrained_model.classifier.in_features, num_classes),
        )
        self.pretrained_model.classifier = self.classifier

    def forward(self, x):
        output = self.pretrained_model(x)
        return output


def test(test_loader, model, device):
    correct_nums_per_g = torch.zeros(4)
    total_nums_per_g = torch.zeros(4)

    with torch.no_grad():
        for i, (images, targets, groups) in enumerate(tqdm(test_loader)):
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

    groups_acc = torch.div(correct_nums_per_g, total_nums_per_g)  # 4 groups acc
    overall_acc = torch.sum(correct_nums_per_g) / torch.sum(total_nums_per_g)  # calculate the overall accuracy

    abs_difference = 0
    minority_acc = groups_acc[3]  # minority group: G10
    for group_acc in groups_acc:
        abs_difference += abs(group_acc - minority_acc)
    SPD = abs_difference / len(groups_acc)
    fairness_score = ((0.2 - SPD) / 0.2)  # calculate the fairness score

    return groups_acc, overall_acc, fairness_score


if __name__ == '__main__':
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # test data transform
    test_transform = albumentations.Compose([
        albumentations.Resize(160, 160),
        albumentations.Normalize(),
        AT.ToTensorV2()
    ])
    # test dataloader
    test_path = "../preprocessed_data/TrainingSet"
    test_loader = DataLoader(MyDataset(test_path, test_transform), batch_size=64,
                             shuffle=True, num_workers=4, pin_memory=True, drop_last=False)

    # my model
    model = MyNet(create_model("mobilenetv3_large_100.ra_in1k"), 6)
    model.to(device)
    model.eval()

    # load model weight
    state_dict = torch.load("../saved_model/mobilenet/mobilenetv3_l-fp16-server-stu-v10.pth", map_location=device)
    new_state_dict = OrderedDict()
    for name, params in state_dict.items():
        if "module" in name:
            name = name[7:]
            new_state_dict[name] = params
        else:
            new_state_dict = state_dict
            break
    del state_dict
    model.load_state_dict(new_state_dict)

    # begin test
    groups_acc, overall_acc, fairness_score = test(test_loader, model, device)
    print("Test Overall Accuracy is", overall_acc.item())
    print("Test accuracy of 4 groups [G6 G7 G8 G10] is", groups_acc.tolist())
    print("Test Fairness score is", fairness_score.item())
