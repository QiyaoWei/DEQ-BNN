import os
import argparse
import torch

from modules.process import train

def get_args():
    parser = argparse.ArgumentParser()
    ############# Multiprocessing #############
    parser.add_argument('--gpu_num', type=int, default=1)
    parser.add_argument('--node_num', type=int, default=1)
    parser.add_argument('--node_rank', type=int, default=0)
    parser.add_argument('--master_addr', type=str, default='127.0.0.1')
    parser.add_argument('--master_port', type=str, default='8855')
    ############# Multiprocessing #############

    ############# Training #############
    parser.add_argument('--train_batch_size', type=int, default=200)
    parser.add_argument('--test_batch_size', type=int, default=200)
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--lr_start', type=float, default=1e-2)
    parser.add_argument('--lr_milestone', type=str, default='150,225')
    parser.add_argument('--test_interval', type=int, default=5)
    parser.add_argument('--seed', type=int, default=777)
    ############# Training #############

    ############# Dataset #############
    parser.add_argument('--dataset', type=str, default='cifar10', choices=['cifar10'])
    parser.add_argument('--download', type=bool, default=True)
    ############# Dataset #############

    ############# DEQ #############
    parser.add_argument('--model', type=str, default='resnet', choices=['resnet', 'deq'])
    parser.add_argument('--solver', type=str, default='anderson', choices=['anderson'])
    parser.add_argument('--tolerance', type=float, default=1e-2)
    ############# DEQ #############

    ############# BNN #############
    # parser.add_argument('--method', type=str, default='VI', choices=['VI'])
    parser.add_argument('--acc_sample_num', type=int, default=8)
    ############# BNN #############

    args = parser.parse_args()
    args.lr_milestone = [int(item) for item in args.lr_milestone.split(',')]
    return args

if __name__=='__main__':
    args = get_args()
    os.environ['NCCL_SOCKET_IFNAME']='lo'
    os.environ['MASTER_ADDR']=args.master_addr
    os.environ['MASTER_PORT']=args.master_port
    torch.multiprocessing.spawn(train, (args, ), nprocs=args.gpu_num*args.node_num)