import argparse

from modules.process import train

def get_args():
    parser = argparse.ArgumentParser()
    ############# GPU #############
    parser.add_argument('--gpu', type=str, default='0')
    ############# GPU #############

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
    parser.add_argument('--download', type=bool, default=False)
    ############# Dataset #############

    ############# DEQ #############
    parser.add_argument('--model', type=str, default='resnet', choices=['resnet', 'deq'])
    parser.add_argument('--solver', type=str, default='anderson', choices=['anderson'])
    parser.add_argument('--tolerance', type=float, default=1e-2)
    ############# DEQ #############

    ############# BNN #############
    # parser.add_argument('--method', type=str, default='VI', choices=['VI'])
    parser.add_argument('--acc_sample_num', type=int, default=20)
    ############# BNN #############

    args = parser.parse_args()
    args.lr_milestone = [int(item) for item in args.lr_milestone.split(',')]
    return args

if __name__=='__main__':
    args = get_args()
    train(args)
