import torch
import os

def save_ckpt(args, model, optimizer, lr_scheduler, epoch, best_performance):
    state = dict(
        model_state=model.state_dict(),
        optimizer=optimizer.state_dict(),
        lr_scheduler=lr_scheduler.state_dict(),
        epoch=epoch,
        best_performance=best_performance,
    )

    torch.save(state, os.path.join(args["ckpt_path"], args["model_name"]+"pth.tar"))


# def load_ckpt(args, model, optimizer, lr_cheduler, epoch)