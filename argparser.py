import torch
import argparse

def args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument('--num_classes', type=int, default=6)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--init_ratio', type=float, default=0.1)
    parser.add_argument('--min_lr_ratio', type=float, default=0.005)
    parser.add_argument('--epochs', type=int, default=150)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--resize', type=int, default=512)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--is_multiscale', type=int, default=0)
    parser.add_argument('--is_parallel', type=int, default=1)
    parser.add_argument('--device_ids', type=list, default=[0,1,2,3])
    parser.add_argument('--optim', default="SGD")
    parser.add_argument('--loss_func', default="CEloss")
    parser.add_argument('--init', default="xavier")
    parser.add_argument('--lr_scheduler', default="Warm-up-Cosine-Annealing")
    parser.add_argument('--model_name',  default="resnet50-fp16-server-mixed-v1")
    parser.add_argument('--train_path', default="../preprocessed_data/TrainingSet")
    parser.add_argument('--val_path',  default="../preprocessed_data/ValSet")
    parser.add_argument('--saved_path', default='../saved_model')
    parser.add_argument('--spd_para', type=float, default=0.2)
    parser.add_argument('--log_dir', default="./log/resnet50/resnet50-fp16-server-mixed-v1")

    args, _ = parser.parse_known_args()
    return args
