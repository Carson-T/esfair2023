import torch
import argparse

def args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument('--num_classes', type=int, default=6)
    parser.add_argument('--lr', type=list, default=[0.0001, 0.0005])
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--init_ratio', type=float, default=0.1)
    parser.add_argument('--min_lr_ratio', type=float, default=0.01)
    parser.add_argument('--drop_rate', type=float, default=0.2)
    parser.add_argument('--drop_path_rate', type=float, default=0)
    parser.add_argument('--epochs', type=int, default=150)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--resize', type=int, default=224)
    parser.add_argument('--num_workers', type=int, default=8)

    parser.add_argument('--is_distill', type=int, default=1)
    parser.add_argument('--teacher_model', default="../saved_model/convnext/convnextv2_n-fp16-server-ext-v5.pth")
    parser.add_argument('--temp', type=int, default=3)
    parser.add_argument('--alpha', type=float, default=0.2)

    parser.add_argument('--is_multiscale', type=int, default=0)
    parser.add_argument('--is_parallel', type=int, default=1)
    parser.add_argument('--use_external', type=int, default=1)
    parser.add_argument('--resume', default="")
    parser.add_argument('--device_ids', type=list, default=[0,1])
    parser.add_argument('--optim', default="AdamW")
    parser.add_argument('--loss_func', default="CEloss")
    parser.add_argument('--init', default="xavier")
    parser.add_argument('--lr_scheduler', default="Warm-up-Cosine-Annealing")
    parser.add_argument('--backbone', default="ghostnet_100.in1k")  # convnextv2_nano.fcmae_ft_in1k  mobilenetv3_large_100.ra_in1k  mobilenetv3_small_100.lamb_in1k
    parser.add_argument('--model_name',  default="ghostnet1.0-fp16-server-stu-v5")
    parser.add_argument('--train_csv_path', default="../preprocessed_data/fold1_train.csv")
    parser.add_argument('--val_csv_path',  default="../preprocessed_data/fold1_val.csv")
    parser.add_argument('--external_csv_path', default="../external_data/external_label.csv")
    parser.add_argument('--saved_path', default='../saved_model/ghostnet')
    parser.add_argument('--ckpt_path', default='../checkpoints/ghostnet')
    parser.add_argument('--spd_para', type=float, default=0.2)
    parser.add_argument('--log_dir', default="./log/ghostnet")

    args, _ = parser.parse_known_args()
    return args
