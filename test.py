import timm
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import pandas as pd
import cv2
import albumentations
from torch.utils.data import DataLoader
from albumentations import pytorch as AT
from torchvision import models
# from model import *
from tqdm import tqdm
import collections
# import torch.ao.quantization.quantize_fx as quantize_fx


# model
class myconvnext(nn.Module):
    def __init__(self, pretrained_model, num_classes):
        super(myconvnext, self).__init__()
        self.pretrained_model = pretrained_model
        self.classifier = nn.Sequential(
            nn.Linear(self.pretrained_model.head.fc.in_features, num_classes),
            # nn.Dropout(0.4),
            # nn.BatchNorm1d(1024),
            # nn.ReLU(),
            # nn.Linear(1024,512),
            # nn.BatchNorm1d(512),
            # nn.ReLU(),
            # nn.Linear(256, num_classes)
        )
        self.pretrained_model.head.fc = self.classifier
        # self.quant = torch.quantization.QuantStub()
        # self.dequant = torch.quantization.DeQuantStub()

    def forward(self, x):
        # x = self.quant(x)
        output = self.pretrained_model(x)
        # output = self.dequant(output)
        return output

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
    model.eval()
    model = model.half()
    with torch.no_grad():
        for i, (images, targets, groups) in enumerate(tqdm(val_loader)):
            groups = groups.to(device).half()
            images = images.to(device).half()
            targets = targets.to(device).half()
            # img_path = list(img_path)
            output = model(images)
            _, preds = torch.max(output, dim=1)
            if i == 0:
                all_preds = preds
                all_targets = targets
                all_groups = groups
                # all_img_paths = img_path
            else:
                all_preds = torch.cat((all_preds, preds))
                all_targets = torch.cat((all_targets, targets))
                all_groups = torch.cat((all_groups, groups))
                # all_img_paths = all_img_paths+img_path
        correct = (all_preds == all_targets)
        for i in range(4):
            correct_per_g = torch.index_select(correct, 0, torch.nonzero(all_groups == i).squeeze())
            total_nums_per_g[i] = correct_per_g.shape[0]
            correct_nums_per_g[i] = torch.sum(correct_per_g).item()

    groups_acc = torch.div(correct_nums_per_g, total_nums_per_g)
    overall_acc = torch.sum(correct_nums_per_g) / torch.sum(total_nums_per_g)  # calculate the overall accuracy

    abs_difference = 0
    minority_acc = groups_acc[3]  # minority group: G10
    for group_acc in groups_acc:
        abs_difference += abs(group_acc - minority_acc)
    SPD = abs_difference / len(groups_acc)
    fairness_score = ((0.2 - SPD) / 0.2)   # calculate the fairness score
    return val_loss / torch.sum(total_nums_per_g), groups_acc, overall_acc, fairness_score



if __name__ == '__main__':
    device = "cuda" if torch.cuda.is_available() else "cpu"

    val_transform = albumentations.Compose([
        albumentations.Resize(480, 480),
        albumentations.Normalize(),
        AT.ToTensorV2()
    ])

    val_loader = DataLoader(MyDataset("../preprocessed_data/fold1_val.csv", val_transform), batch_size=64,
                            shuffle=True, num_workers=8, pin_memory=True, drop_last=False)
    pretrained_model = timm.create_model("convnextv2_nano.fcmae_ft_in1k")
    model = myconvnext(pretrained_model, 6)
    model.to(device)
    state_dict = torch.load("../saved_model/convnext/convnextv2_n-fp16-server-ext-v2.pth", map_location=device)
    new_state_dict = collections.OrderedDict()
    for name, params in state_dict.items():
        name = name[7:]
        new_state_dict[name] = params
    del state_dict
    model.load_state_dict(new_state_dict)
    # example_inputs = torch.randn(64,3,480,480).to(device)
    # qconfig_mapping = torch.ao.quantization.get_default_qconfig_mapping('qnnpack')
    # prepare
    # model_prepared = quantize_fx.prepare_fx(model, qconfig_mapping, example_inputs)
    # calibrate(model_prepared,val_loader)
    # model_int8 = quantize_fx.convert_fx(model_prepared)

    _, groups_acc, overall_acc, fairness_score = test(val_loader, model, device)
    print(overall_acc, groups_acc)
    # result = {"img_path":[], "group":[]}
    # for i in range(len(all_targets)):
    #     if all_targets[i] != all_preds[i]:
    #         result["img_path"].append(all_img_paths[i])
    #         result["group"].append((all_groups[i]))

    # pd.DataFrame(result).to_csv("../result.csv",index=False)
    # torch.onnx.export(model, img, "./onnx/resnet50-v1.onnx")