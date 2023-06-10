import torch
from torch.cuda.amp import autocast
from tqdm import *
#calculate the number of correct predictions
# def num_correct_pred(output, target, class_num=None):
#     if class_num != None:
#         mask = (target == class_num)
#         output = output[mask]
#         target = target[mask]
#
#     _, pred = torch.max(output, dim=1)
#     correct_predictions = torch.sum(pred == target).item()
#
#     return correct_predictions

#training
def train(train_loader, model, criterion, optimizer, scaler, args):
    model.train()
    correct_nums_per_g = torch.zeros(4)
    total_nums_per_g = torch.zeros(4)
    training_loss = 0.0
    for i,(images, targets, groups) in enumerate(tqdm(train_loader)):
        groups = groups.to(args["device"])
        images = images.to(args["device"])
        targets = targets.to(args["device"])
        with autocast():
            output = model(images)
            loss = criterion(output, targets)
        _, pred = torch.max(output, dim=1)
        training_loss += loss.item()
        correct = (pred == targets)
        for i in range(4):
            correct_per_g = torch.index_select(correct, 0, torch.nonzero(groups == i).squeeze())
            total_nums_per_g[i] += correct_per_g.shape[0]
            correct_nums_per_g[i] += torch.sum(correct_per_g).item()

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
    groups_acc = torch.div(correct_nums_per_g, total_nums_per_g)
    overall_acc = torch.sum(correct_nums_per_g) / torch.sum(total_nums_per_g) #calculate the overall accuracy
    return training_loss, overall_acc, groups_acc

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
            _, pred = torch.max(output, dim=1)
            val_loss += loss.item()
            correct = (pred == targets)
            for i in range(4):
                correct_per_g = torch.index_select(correct, 0, torch.nonzero(groups == i).squeeze())
                total_nums_per_g[i] += correct_per_g.shape[0]
                correct_nums_per_g[i] += torch.sum(correct_per_g).item()

    groups_acc = torch.div(correct_nums_per_g, total_nums_per_g)
    overall_acc = torch.sum(correct_nums_per_g) / torch.sum(total_nums_per_g)  # calculate the overall accuracy
    return val_loss, overall_acc, groups_acc



