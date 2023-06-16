import timm
import torch
from torch.utils.data import Dataset
import pandas as pd
import cv2
import albumentations
from torch.utils.data import DataLoader
from albumentations import pytorch as AT
from torchvision import models
from model import *
from tqdm import tqdm


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
        return img, label, group, img_path

    def _make_dataset(self):
        data = pd.read_csv(self.csv_path)
        img_paths = data["path"].values.tolist()
        labels = [self.class_dict[i] for i in data["label"].values]
        groups = [self.group_dict[i] for i in data["group"].values]

        return img_paths, labels, groups

def test(val_loader, model, device):
    model.eval()
    correct_nums_per_g = torch.zeros(4)
    total_nums_per_g = torch.zeros(4)
    val_loss = 0.0
    model = model
    with torch.no_grad():
        for i, (images, targets, groups, img_path) in enumerate(tqdm(val_loader)):
            groups = groups.to(device)
            images = images.to(device)
            targets = targets.to(device)
            img_path = list(img_path)
            output = model(images)
            _, preds = torch.max(output, dim=1)
            if i == 0:
                all_preds = preds
                all_targets = targets
                all_groups = groups
                all_img_paths = img_path
            else:
                all_preds = torch.cat((all_preds, preds))
                all_targets = torch.cat((all_targets, targets))
                all_groups = torch.cat((all_groups, groups))
                all_img_paths = all_img_paths+img_path
        correct = (all_preds == all_targets)
        for i in range(4):
            correct_per_g = torch.index_select(correct, 0, torch.nonzero(all_groups == i).squeeze())
            total_nums_per_g[i] = correct_per_g.shape[0]
            correct_nums_per_g[i] = torch.sum(correct_per_g).item()

    groups_acc = torch.div(correct_nums_per_g, total_nums_per_g)
    overall_acc = torch.sum(correct_nums_per_g) / torch.sum(total_nums_per_g)  # calculate the overall accuracy
    return val_loss / torch.sum(total_nums_per_g), overall_acc, groups_acc, all_preds.cpu(), all_targets.cpu(), all_groups.cpu(), all_img_paths


if __name__ == '__main__':
    device = "cuda" if torch.cuda.is_available() else "cpu"

    val_transform = albumentations.Compose([
        albumentations.Resize(480, 480),
        albumentations.Normalize(),
        AT.ToTensorV2()
    ])
    val_loader = DataLoader(MyDataset("../preprocessed_data/fold1_val.csv", val_transform), batch_size=4,
                            shuffle=True, num_workers=2, pin_memory=True, drop_last=False)
    pretrained_model = timm.create_model("efficientnetv2_rw_s.ra2_in1k")
    # pretrained_model = models.resnet101(pretrained=False)
    model = efficientnet(pretrained_model, 6)
    model.to(device)
    model = nn.DataParallel(model, device_ids=[0, 1])

    model.load_state_dict(
        torch.load("../saved_model/efficientNet/effnetv2_s-fp16-server-mixed-v4.pth", map_location=device))

    _, overall_acc, groups_acc, all_preds, all_targets, all_groups, all_img_paths = test(val_loader, model, device)
    print(overall_acc, groups_acc)
    result = {"img_path":[], "group":[]}
    for i in range(len(all_targets)):
        if all_targets[i] != all_preds[i]:
            result["img_path"].append(all_img_paths[i])
            result["group"].append((all_groups[i]))

    pd.DataFrame(result).to_csv("../result.csv",index=False)
    # torch.onnx.export(model, img, "./onnx/resnet50-v1.onnx")