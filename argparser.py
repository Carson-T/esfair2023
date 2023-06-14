import torch
import argparse

def args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument('--num_classes', type=int, default=6)
    parser.add_argument('--lr', type=float, default=0.00001)
    parser.add_argument('--weight_decay', type=float, default=0.05)
    parser.add_argument('--init_ratio', type=float, default=0.1)
    parser.add_argument('--min_lr_ratio', type=float, default=0.01)
    parser.add_argument('--epochs', type=int, default=150)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--resize', type=int, default=480)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--is_multiscale', type=int, default=0)
    parser.add_argument('--is_parallel', type=int, default=1)
    parser.add_argument('--device_ids', type=list, default=[0,1])
    parser.add_argument('--optim', default="AdamW")
    parser.add_argument('--loss_func', default="CEloss")
    parser.add_argument('--init', default="xavier")
    parser.add_argument('--lr_scheduler', default="Warm-up-Cosine-Annealing")
    parser.add_argument('--backbone', default="convnextv2_nano.fcmae_ft_in1k")
    parser.add_argument('--model_name',  default="convnextv2_n-fp16-server-mixed-v2")
    parser.add_argument('--train_csv_path', default="../preprocessed_data/fold1_train.csv")
    parser.add_argument('--val_scv_path',  default="../preprocessed_data/fold1_val.csv")
    parser.add_argument('--saved_path', default='../saved_model/convnext')
    parser.add_argument('--spd_para', type=float, default=0.2)
    parser.add_argument('--log_dir', default="./log/convnext/convnextv2_n-fp16-server-mixed-v2")

    args, _ = parser.parse_known_args()
    return args
