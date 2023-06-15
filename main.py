from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler
import torch.backends.cudnn as cudnn
from torchvision import models
import os
import json
import random
import numpy as np
import timm
from torch.utils.tensorboard import SummaryWriter
from train import *
from dataset import MyDataset
from argparser import args_parser
from model import *
from utils.initialize import *
from utils.FocalLoss import FocalLoss
from utils.confusion_matrix import plot_matrix
from utils.transform import transform
import models.inceptionnext

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
    train_transform, val_transform = transform(args)

    train_loader = DataLoader(MyDataset(args["train_csv_path"], train_transform), batch_size=args["batch_size"],
                              shuffle=True, num_workers=args["num_workers"], pin_memory=True, drop_last=True)
    val_loader = DataLoader(MyDataset(args["val_csv_path"], val_transform), batch_size=args["batch_size"],
                            shuffle=True, num_workers=args["num_workers"], pin_memory=True, drop_last=True)
    if args["is_parallel"] == 1:
        model = nn.DataParallel(model, device_ids=args["device_ids"])
    model.to(args["device"])
    if args["init"] == "xavier":
        model.apply(xavier)
    elif args["init"] == "kaiming":
        model.apply(kaiming)

    if args["optim"] == "AdamW":
        optimizer = torch.optim.AdamW(model.parameters(), lr=args["lr"], weight_decay=args["weight_decay"])
    elif args["optim"] == "SGD":
        optimizer = torch.optim.SGD(model.parameters(), lr=args["lr"], momentum=0.9, weight_decay=args["weight_decay"])

    if args["loss_func"] == "CEloss":
        # weight = torch.tensor([0.033, 0.041, 0.026, 0.02, 0.008, 0.872])
        weight = 5207 * 1 / torch.tensor([1231, 982, 1537, 2206, 5207, 49])
        loss_func = torch.nn.CrossEntropyLoss(weight).to(args["device"])
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

    nums2groups = {0: "G6", 1: "G7", 2: "G8", 3: "G10"}
    performance_score_init = 0
    for iter in range(1, args["epochs"] + 1):
        train_loss, train_overall_acc, train_groups_acc = train(train_loader, model, loss_func, optimizer, scaler, args)
        val_loss, val_overall_acc, val_groups_acc, all_preds, all_targets = val(val_loader, model, loss_func, args)
        for i in range(4):
            print(f'Epoch {iter}: group ' + nums2groups[i] + f' val acc: {val_groups_acc[i]}')
            writer.add_scalars("groups acc/" + "acc of " + nums2groups[i], {"train_acc": train_groups_acc[i],
                                                                            "val_acc": val_groups_acc[i]}, iter)
        print(f'Epoch {iter}: train overall acc: {train_overall_acc}')
        print(f'Epoch {iter}: val overall acc: {val_overall_acc}')
        writer.add_scalars("loss/" + "overall_loss", {"train_loss": train_loss, "val_loss": val_loss}, iter)
        lr_scheduler.step()

        abs_difference = 0
        minority_acc = val_groups_acc[3]  # minority group: G10
        for group_acc in val_groups_acc:
            abs_difference += abs(group_acc - minority_acc)
        SPD = abs_difference / len(val_groups_acc)
        fairness = ((args["spd_para"] - SPD) / args["spd_para"])

        acc_score = val_overall_acc / 3
        fairness_score = fairness / 3
        performance_score = acc_score + fairness_score
        print(f"Epoch {iter}, SPD: {SPD}")
        print(f"Epoch {iter}, Fairness_score: {fairness_score}")
        print(f"Epoch {iter}, Accuracy score: {acc_score}")
        print(f"Epoch {iter}, Performance score: {performance_score}")
        writer.add_scalars("overall",
                           {"overall_acc": val_overall_acc, "fairness": fairness,
                            "performance_score": performance_score},
                           iter)
        if performance_score > performance_score_init:
            performance_score_init = performance_score
            plot_matrix(all_targets.cpu(), all_preds.cpu(), [0, 1, 2, 3, 4, 5], args["log_dir"] + "/confusion_matrix.jpg",
                        ['BCC', 'BKL', 'MEL', 'NV', 'unknown', 'VASC'])
            torch.save(model.state_dict(), args["saved_path"] + "/" + args["model_name"] + ".pth")


if __name__ == '__main__':
    args = vars(args_parser())
    set_seed(2023)
    pretrained_model = timm.create_model(args["backbone"], pretrained=True)
    # pretrained_model = models.resnet50(pretrained=True)
    if "resnet" in args["backbone"]:
        model = resnet(pretrained_model, args["num_classes"])
    elif "efficientnet" in args["backbone"]:
        model = efficientnet(pretrained_model, args["num_classes"])
    elif "convnext" in args["backbone"]:
        model = myconvnext(pretrained_model, args["num_classes"])
    elif "inceptionnext" in args["backbone"]:
        model = InceptionNext(pretrained_model, args["num_classes"])
    main(args, model)
    with open(args["log_dir"] + "/parameters.json", "w+") as f:
        json.dump(args, f)
