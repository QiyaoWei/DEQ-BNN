from __future__ import print_function

import os
import argparse

import torch
import numpy as np
from torch.optim import Adam, lr_scheduler
from torch.nn import functional as F

import data
import metric
from model import DEQFixedPoint, BBBLeNet, anderson

# Hyperparams
layer_type = 'bbb'  # 'bbb' or 'lrt'
activation_type = 'softplus'  # 'softplus' or 'relu'
priors={
    'prior_mu': 0,
    'prior_sigma': 0.1,
    'posterior_mu_initial': (0, 0.1),  # (mean, std) normal_
    'posterior_rho_initial': (-5, 0.1),  # (mean, std) normal_
}
n_epochs = 200
lr_start = 0.001
num_workers = 4
valid_size = 0.2
batch_size = 256
train_ens = 1
valid_ens = 1
beta_type = 0.1  # 'Blundell', 'Standard', etc. Use float for const value

# CUDA settings
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def getModel(net_type, inputs, outputs, priors, layer_type, activation_type):
    if (net_type == 'lenet'):
        return BBBLeNet(outputs, inputs, priors, layer_type, activation_type)
#         return DEQFixedPoint(f, anderson, tol=1e-4, max_iter=100, beta=2.0)
    elif (net_type == 'alexnet'):
        return BBBAlexNet(outputs, inputs, priors, layer_type, activation_type)
    elif (net_type == '3conv3fc'):
        return BBB3Conv3FC(outputs, inputs, priors, layer_type, activation_type)
    else:
        raise ValueError('Network should be either [LeNet / AlexNet / 3Conv3FC')


def logmeanexp(x, dim=None, keepdim=False):
    """Stable computation of log(mean(exp(x))"""

    
    if dim is None:
        x, dim = x.view(-1), 0
    x_max, _ = torch.max(x, dim, keepdim=True)
    x = x_max + torch.log(torch.mean(torch.exp(x - x_max), dim, keepdim=True))
    return x if keepdim else x.squeeze(dim)


def train_model(net, optimizer, criterion, trainloader, num_ens=1, beta_type=0.1, epoch=None, num_epochs=None):
    net.train()
    training_loss = 0.0
    accs = []
    kl_list = []
    for i, (inputs, labels) in enumerate(trainloader, 1):

        optimizer.zero_grad()

        inputs, labels = inputs.to(device), labels.to(device)
        outputs = torch.zeros(inputs.shape[0], 10, num_ens).to(device)

        kl = 0.0
        for j in range(num_ens):
            net_out, _kl = net(inputs)
#             print(net_out)
#             print(_kl)
            kl += _kl
            outputs[:, :, j] = F.log_softmax(net_out, dim=1)
        
        kl = kl / num_ens
        kl_list.append(kl.item())
        log_outputs = logmeanexp(outputs, dim=2)

        beta = metric.get_beta(i-1, len(trainloader), beta_type, epoch, num_epochs)
        loss = criterion(log_outputs, labels, kl, beta)
#         print(kl)
#         print(loss)
        loss.backward()
        optimizer.step()

        accs.append(metric.acc(log_outputs.data, labels))
        training_loss += loss.cpu().data.numpy()
#         print(accs)
#         print(training_loss)
    return training_loss/len(trainloader), np.mean(accs), np.mean(kl_list)


def validate_model(net, criterion, validloader, num_ens=1, beta_type=0.1, epoch=None, num_epochs=None):
    """Calculate ensemble accuracy and NLL Loss"""
    net.train()
    valid_loss = 0.0
    accs = []

    for i, (inputs, labels) in enumerate(validloader):
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = torch.zeros(inputs.shape[0], net.num_classes, num_ens).to(device)
        kl = 0.0
        for j in range(num_ens):
            net_out, _kl = net(inputs)
            kl += _kl
            outputs[:, :, j] = F.log_softmax(net_out, dim=1).data

        log_outputs = logmeanexp(outputs, dim=2)

        beta = metric.get_beta(i-1, len(validloader), beta_type, epoch, num_epochs)
        valid_loss += criterion(log_outputs, labels, kl, beta).item()
        accs.append(metric.acc(log_outputs, labels))

    return valid_loss/len(validloader), np.mean(accs)


def run(dataset, net_type):

    trainset, testset, inputs, outputs = data.getDataset(dataset)
    train_loader, valid_loader, test_loader = data.getDataloader(
        trainset, testset, valid_size, batch_size, num_workers)
    net = getModel(net_type, inputs, outputs, priors, layer_type, activation_type).to(device)

    ckpt_dir = f'checkpoints/{dataset}/bayesian'
    ckpt_name = f'checkpoints/{dataset}/bayesian/model_{net_type}_{layer_type}_{activation_type}.pt'

    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir, exist_ok=True)

    criterion = metric.ELBO(len(trainset)).to(device)
    optimizer = Adam(net.parameters(), lr=lr_start)
    lr_sched = lr_scheduler.ReduceLROnPlateau(optimizer, patience=6, verbose=True)
    valid_loss_max = np.Inf
    for epoch in range(n_epochs):  # loop over the dataset multiple times

        train_loss, train_acc, train_kl = train_model(net, optimizer, criterion, train_loader, num_ens=train_ens, beta_type=beta_type, epoch=epoch, num_epochs=n_epochs)
        valid_loss, valid_acc = validate_model(net, criterion, valid_loader, num_ens=valid_ens, beta_type=beta_type, epoch=epoch, num_epochs=n_epochs)
        lr_sched.step(valid_loss)

        print('Epoch: {} \tTraining Loss: {:.4f} \tTraining Accuracy: {:.4f} \tValidation Loss: {:.4f} \tValidation Accuracy: {:.4f} \ttrain_kl_div: {:.4f}'.format(
            epoch, train_loss, train_acc, valid_loss, valid_acc, train_kl))

        # save model if validation accuracy has increased
        if valid_loss <= valid_loss_max:
            print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(
                valid_loss_max, valid_loss))
            torch.save(net.state_dict(), ckpt_name)
            valid_loss_max = valid_loss

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = "PyTorch Bayesian Model Training")
    parser.add_argument('--net_type', default='lenet', type=str, help='model')
    parser.add_argument('--dataset', default='CIFAR10', type=str, help='dataset = [MNIST/CIFAR10/CIFAR100]')
    args = parser.parse_args()

    run(args.dataset, args.net_type)