import torch
import albumentations
from torch.utils.data import DataLoader
from dataset import MyDataset
from albumentations import pytorch as AT
from torchvision import models
from model import resnet
from tqdm import tqdm


def test(val_loader, model, device):
    model.eval()
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
                all_preds = torch.cat((all_preds,preds))
                all_targets = torch.cat((all_targets,targets))
                all_groups = torch.cat((all_groups,groups))
        correct = (all_preds == all_targets)
        for i in range(4):
            correct_per_g = torch.index_select(correct, 0, torch.nonzero(all_groups == i).squeeze())
            total_nums_per_g[i] = correct_per_g.shape[0]
            correct_nums_per_g[i] = torch.sum(correct_per_g).item()

    groups_acc = torch.div(correct_nums_per_g, total_nums_per_g)
    overall_acc = torch.sum(correct_nums_per_g) / torch.sum(total_nums_per_g)  # calculate the overall accuracy
    return val_loss / torch.sum(total_nums_per_g), overall_acc, groups_acc, all_preds, all_targets

if __name__ == '__main__':
    device = "cuda" if torch.cuda.is_available() else "cpu"

    val_transform = albumentations.Compose([
        albumentations.Resize(512, 512),
        albumentations.Normalize(),
        AT.ToTensorV2()
    ])
    val_loader = DataLoader(MyDataset("../preprocessed_data/ValSet", val_transform), batch_size=4,
                            shuffle=True, num_workers=2, pin_memory=True, drop_last=False)
    pretrained_model = models.resnet101(pretrained=True)
    model = resnet(pretrained_model, 6)


    model.load_state_dict(torch.load('../saved_model/resnet101-fp16-cloud-mixed-v1.pth',map_location=device))
    model.to(device)

    _, overall_acc, groups_acc, _, _ = test(val_loader,model,device)


    # torch.onnx.export(model, img, "./onnx/resnet50-v1.onnx")