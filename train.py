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
    correct_predictions = 0
    total_targets = 0
    training_loss = 0.0
    for i,(images, targets) in enumerate(tqdm(train_loader)):
        total_targets += targets.shape[0]
        images = images.to(args["device"])
        targets = targets.to(args["device"])
        with autocast():
            output = model(images)
            loss = criterion(output, targets)
        _, pred = torch.max(output, dim=1)
        training_loss += loss.item()
        correct_predictions += torch.sum(pred == targets).item()

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

    overall_acc = correct_predictions / total_targets #calculate the overall accuracy
    return training_loss, overall_acc, correct_predictions, total_targets

def val(val_loader, model, criterion, args):
    model.eval()
    correct_predictions = 0
    total_targets = 0
    val_loss = 0.0
    with torch.no_grad():
        for i, (images, targets) in enumerate(tqdm(val_loader)):
            total_targets += targets.shape[0]
            images = images.to(args["device"])
            targets = targets.to(args["device"])
            # with autocast():
            output = model(images)
            loss = criterion(output, targets)
            _, pred = torch.max(output, dim=1)
            val_loss += loss.item()
            correct_predictions += torch.sum(pred == targets).item()

    overall_acc = correct_predictions / total_targets  # calculate the overall accuracy
    return val_loss, overall_acc, correct_predictions, total_targets



