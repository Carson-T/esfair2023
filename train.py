import torch
from torch.cuda.amp import autocast
import torch.nn.functional as F
from tqdm import *


# training
def train(train_loader, model, criterion, optimizer, scaler, args):
    model.train()
    correct_nums_per_g = torch.zeros(4)
    total_nums_per_g = torch.zeros(4)
    training_loss = 0.0
    for i, (images, targets, groups) in enumerate(tqdm(train_loader)):
        groups = groups.to(args["device"])
        images = images.to(args["device"])
        targets = targets.to(args["device"])
        with autocast():
            output = model(images)
            loss = criterion(output, targets)
        _, preds = torch.max(output, dim=1)
        training_loss += loss.item()
        if i == 0:
            all_preds = preds
            all_targets = targets
            all_groups = groups
        else:
            all_preds = torch.cat((all_preds, preds))
            all_targets = torch.cat((all_targets, targets))
            all_groups = torch.cat((all_groups, groups))

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

    correct = (all_preds == all_targets)
    for i in range(4):
        correct_per_g = torch.index_select(correct, 0, torch.nonzero(all_groups == i).squeeze())
        total_nums_per_g[i] = correct_per_g.shape[0]
        correct_nums_per_g[i] = torch.sum(correct_per_g).item()

    groups_acc = torch.div(correct_nums_per_g, total_nums_per_g)
    overall_acc = torch.sum(correct_nums_per_g) / torch.sum(total_nums_per_g)  # calculate the overall accuracy
    return training_loss / torch.sum(total_nums_per_g), overall_acc, groups_acc


def val(val_loader, model, criterion, args):
    model.eval()
    correct_nums_per_g = torch.zeros(4)
    total_nums_per_g = torch.zeros(4)
    val_loss = 0.0
    with torch.no_grad():
        for i, (images, targets, groups) in enumerate(tqdm(val_loader)):
            groups = groups.to(args["device"])
            images = images.to(args["device"])
            targets = targets.to(args["device"])
            # with autocast():
            output = model(images)
            loss = criterion(output, targets)
            _, preds = torch.max(output, dim=1)
            val_loss += loss.item()
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

    groups_acc = torch.div(correct_nums_per_g, total_nums_per_g)
    overall_acc = torch.sum(correct_nums_per_g) / torch.sum(total_nums_per_g)  # calculate the overall accuracy
    return val_loss / torch.sum(total_nums_per_g), overall_acc, groups_acc, all_preds, all_targets


def external_train(train_loader, model, criterion, optimizer, scaler, args):
    model.train()
    training_loss = 0.0
    for i, (images, targets, groups) in enumerate(tqdm(train_loader)):
        images = images.to(args["device"])
        targets = targets.to(args["device"])
        with autocast():
            output = model(images)
            loss = criterion(output, targets)
        _, preds = torch.max(output, dim=1)
        training_loss += loss.item()

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

def distill_train(train_loader, student_model, teacher_model, hard_loss, soft_loss, optimizer, scaler, args):
    student_model.train()
    correct_nums_per_g = torch.zeros(4)
    total_nums_per_g = torch.zeros(4)
    training_loss = 0.0
    for i, (images, targets, groups) in enumerate(tqdm(train_loader)):
        groups = groups.to(args["device"])
        images = images.to(args["device"])
        targets = targets.to(args["device"])
        with torch.no_grad():
            teacher_output = teacher_model(images)
        with autocast():
            student_output = student_model(images)
            loss1 = hard_loss(student_output, targets)
            loss2 = soft_loss(
                F.softmax(student_output / args["temp"], dim=1),
                F.softmax(teacher_output / args["temp"], dim=1)
            )
            loss = args["alpha"] * loss1 + (1 - args["alpha"]) * loss2
        _, preds = torch.max(teacher_output, dim=1)
        training_loss += loss.item()
        if i == 0:
            all_preds = preds
            all_targets = targets
            all_groups = groups
        else:
            all_preds = torch.cat((all_preds, preds))
            all_targets = torch.cat((all_targets, targets))
            all_groups = torch.cat((all_groups, groups))

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

    correct = (all_preds == all_targets)
    for i in range(4):
        correct_per_g = torch.index_select(correct, 0, torch.nonzero(all_groups == i).squeeze())
        total_nums_per_g[i] = correct_per_g.shape[0]
        correct_nums_per_g[i] = torch.sum(correct_per_g).item()

    groups_acc = torch.div(correct_nums_per_g, total_nums_per_g)
    overall_acc = torch.sum(correct_nums_per_g) / torch.sum(total_nums_per_g)  # calculate the overall accuracy
    return training_loss / torch.sum(total_nums_per_g), overall_acc, groups_acc