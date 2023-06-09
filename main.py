import torch
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler
import torch.backends.cudnn as cudnn
from torchvision import models
import os
import json
import random
import numpy as np
import torch.nn as nn
import timm
import albumentations
from albumentations import pytorch as AT
from torch.utils.tensorboard import SummaryWriter
from train import *
from dataset import MyDataset
from argparser import args_parser
from model import *
from utils.initialize import *
from utils.FocalLoss import FocalLoss


def set_seed(seed=2023):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cudnn.benchmark = True
    cudnn.deterministic = True
    os.environ['PYTHONHASHSEED'] = str(seed)


def main(args, model):
    writer = SummaryWriter(log_dir=args["log_dir"])
    scaler = GradScaler()
    groups = ["G6", "G7", "G8", "G10"]

    if args["is_multiscale"] == 1:
        train_resize = albumentations.OneOf([
            albumentations.Resize(512, 512),
            albumentations.Resize(450, 450),
            albumentations.Resize(400, 400),
            albumentations.Resize(370, 370),
        ], p=1)
        val_resize = albumentations.Resize(512, 512)
    else:
        train_resize = val_resize = albumentations.Resize(args["resize"], args["resize"])

    train_transform = albumentations.Compose([
        train_resize,
        # albumentations.CenterCrop(64, 64),
        albumentations.GaussNoise(p=0.5),
        albumentations.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.5),
        albumentations.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        albumentations.ShiftScaleRotate(scale_limit=0.2, rotate_limit=90, p=0.5),
        albumentations.Flip(p=0.5),
        albumentations.PadIfNeeded(512, 512),
        albumentations.Normalize(),
        AT.ToTensorV2()
    ])

    val_transform = albumentations.Compose([
        val_resize,
        albumentations.Normalize(),
        AT.ToTensorV2()
    ])

    train_sets = []
    val_sets = []
    # put DataLoader with different groups
    for i in range(len(groups)):
        print(groups[i])
        train_sets.append(DataLoader(MyDataset(os.path.join(args["train_path"], groups[i]), train_transform),
                                     batch_size=args["batch_size"], shuffle=True, num_workers=args["num_workers"], pin_memory=True,
                                     drop_last=True))
        val_sets.append(DataLoader(MyDataset(os.path.join(args["val_path"], groups[i]), val_transform),
                                   batch_size=args["batch_size"], shuffle=True, num_workers=args["num_workers"], pin_memory=True,
                                   drop_last=True))
    if args["is_parallel"] == 1:
        model = nn.DataParallel(model,device_ids=args["device_ids"])
    model.to(args["device"])
    if args["init"] == "xavier":
        model.apply(xavier)
    elif args["init"] == "kaiming":
        model.apply(kaiming)

    if args["optim"] == "Adam":
        optimizer = torch.optim.Adam(model.parameters(),lr=args["lr"])
    elif args["optim"] == "SGD":
        optimizer = torch.optim.SGD(model.parameters(), lr=args["lr"], momentum=0.9, weight_decay=args["weight_decay"])

    if args["loss_func"] == "CEloss":
        loss_func = torch.nn.CrossEntropyLoss(torch.tensor([0.033, 0.041, 0.026, 0.02, 0.008, 0.872])).to(
            args["device"])
    elif args["loss_func"] == "FocalLoss":
        loss_func = FocalLoss(alpha=[0.033, 0.041, 0.026, 0.02, 0.008, 0.872]).to(args["device"])

    if args["lr_scheduler"] == "Warm-up-Cosine-Annealing":
        init_ratio, warm_up_steps, min_lr_ratio, max_steps = args["init_ratio"], args["epochs"] / 10, args[
            "min_lr_ratio"], args["epochs"]
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda step: \
            (1 - init_ratio) / (warm_up_steps - 1) * step + init_ratio if step < warm_up_steps - 1 \
                else (1 - min_lr_ratio) * 0.5 * (np.cos(
                (step - (warm_up_steps - 1)) / (max_steps - (warm_up_steps - 1)) * np.pi) + 1) + min_lr_ratio)
    elif args["lr_scheduler"] == "ReduceLROnPlateau":
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                                  factor=0.5, patience=10, verbose=True,
                                                                  min_lr=args["min_lr_ratio"] * args["lr"])

    # lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda step: \
    #                     (1 - init_ratio) / (warm_up_steps - 1) * step + init_ratio if step < warm_up_steps - 1 \
    #                     else (1 - min_lr_ratio) * ((max_steps-step)/(max_steps-(warm_up_steps-1))) + min_lr_ratio)
    val_groups_acc_dict = {}
    performance_score_init = 0
    for iter in range(1, args["epochs"] + 1):
        val_correct_num = 0
        val_total_num = 0
        for i in range(4):
            # get result after training
            train_loss, train_acc, train_correct_num_per_group, train_total_num_per_group = train(train_sets[i], model,
                                                                                                  loss_func, optimizer,
                                                                                                  scaler, args)
            val_loss, val_acc, val_correct_num_per_group, val_total_num_per_group = val(val_sets[i], model, loss_func,
                                                                                        args)
            val_groups_acc_dict[groups[i]] = val_acc
            val_correct_num += val_correct_num_per_group
            val_total_num += val_total_num_per_group
            print(f'Epoch {iter}: Acc for Group {groups[i]}: {val_acc}')
            writer.add_scalars("groups acc/" + "acc of " + groups[i], {"train_acc": train_acc, "val_acc": val_acc},
                               iter)
            writer.add_scalars("groups loss/" + "loss of " + groups[i],
                               {"train_loss": train_loss / train_total_num_per_group,
                                "val_loss": val_loss / val_total_num_per_group}, iter)
        lr_scheduler.step()
        overall_acc = val_correct_num / val_total_num
        print(f"Epoch {iter}, Overall_Acc: {overall_acc}")
        abs_difference = 0
        minority_acc = val_groups_acc_dict['G10']  # minority group: G10
        for G, group_acc in val_groups_acc_dict.items():
            abs_difference += abs(group_acc - minority_acc)
        SPD = abs_difference / len(val_groups_acc_dict)
        fairness = ((args["spd_para"] - SPD) / args["spd_para"])

        acc_score = overall_acc / 3
        fairness_score = fairness / 3
        performance_score = acc_score + fairness_score
        print(f"Epoch {iter}, SPD: {SPD}")
        print(f"Epoch {iter}, Fairness_score: {fairness_score}")
        print(f"Epoch {iter}, Accuracy score: {acc_score}")
        print(f"Epoch {iter}, Performance score: {performance_score}")
        writer.add_scalars("overall",
                           {"overall_acc": overall_acc, "fairness": fairness, "performance_score": performance_score},
                           iter)
        if performance_score > performance_score_init:
            performance_score_init = performance_score
            torch.save(model.state_dict(), args["saved_path"] + "/" + args["model_name"] + ".pth")


if __name__ == '__main__':
    args = vars(args_parser())
    set_seed(2023)
    pretrained_model = timm.create_model("resnext50_32x4d", pretrained=True)
    # pretrained_model = models.resnet50(pretrained=True)
    model = resnet50(pretrained_model, args["num_classes"])
    main(args, model)
    with open(args["log_dir"]+"/parameters.json","w+") as f:
        json.dump(args, f)
