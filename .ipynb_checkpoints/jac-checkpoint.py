# gt = {'0':1296, '1':48, '2':48, '3':48, '4':27648, '5':27648, '6':64, '7':64, '8':48, '9':48, '10':48, '11':48, '12':48, '13':48, '14':7680, '15':10}
# TODO list:
# 1. Is vjp twice faster than functional jacobian? (https://discuss.pytorch.org/t/computing-vector-jacobian-and-jacobian-vector-product-efficiently/69595/5)
# 2. We are currently linearizing the entire model. Can we linearize only the DEQ? That means the linear layer will not feature in our calculations and only function as a forward/backward pass
# 3. There are mixed uses of einsum and @ in the code. Should they be unified?



import time
from collections import OrderedDict
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
torch.autograd.set_detect_anomaly(True)
from modules.model import ResDEQ
import re
from torch.utils.data import DataLoader
from modules.datatool import get_dataset
from scipy.linalg import eigh
import math
from functorch import make_functional_with_buffers, vmap, vjp, jvp, jacrev
import torch
from torch.autograd.functional import jacobian, hessian
from torch.nn.utils import _stateless
torch.manual_seed(6666)


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

if __name__ == '__main__':

    model = ResDEQ(3, 48, 64, 10).float().to(device)
#     model.load_state_dict(torch.load("load/model.pth"))
    model.eval()
    num_params = count_parameters(model)
    names = list(n for n, _ in model.named_parameters())

    # This is the nystrom parameter. Namely, how many data samples are we taking?
    # There are actually two hyperparameters in Nystrom: the number of Monte Carlo samples, and the number of top eigenvalues
    # Right now the assumption is that those two are the same, both taken to be 100
    nystrom_samples = 100
    # Note that nystrom currently takes the first x training data from train_loader
    # Therefore, it is necessary that batch_size > nystrom_samples, which is a reasonable assumption
    batch_size = 200
    train_set, test_set, in_channels, num_classes = get_dataset('cifar10', True)
    train_loader = DataLoader(train_set, batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size, shuffle=False)
    # Declar eigfuns here for later use
    eigfuns = None


    # This is the kernel wrt which we calculated its nystrom
    def ntk_kernel(net, jac1, jac2=None, saved_kernel=None):
        if jac2 == None:

            # This just collects the jacobian tuple into a tensor
            return torch.cat([torch.permute(t.reshape(nystrom_samples, 10, -1), (1, 0, 2)) for t in jac1], dim=2)

        else:
            # Here we are doing inference using the nystrom kernel
            # input data is jac1, and we don't need jac2 because that info is in saved_kernel
            assert saved_kernel != None
            final_jac = torch.cat([torch.permute(t.reshape(batch_size, 10, -1), (1, 0, 2)) for t in jac1], dim=2)
            # TODO: find out whether torch.einsum('bij,bkj->bik', final_jac, saved_kernel).to(device) is more efficient
            return torch.stack([final_jac[i] @ saved_kernel[i].T for i in range(len(final_jac))])

    # The only notable change in this function is that X is passed in NOT as raw data, but jacobian
    # This is so that we don't have to re-calculate the jacobian every time
    def nystrom(net, X, kernel):
        start = time.time()
        saved_kernel = kernel(net, X)
        torch.save(saved_kernel, "ker.pt")
        # TODO: find out whether torch.einsum('bij,bkj->bik', saved_kernel, saved_kernel).to(device) is more efficient
        nystrom_kernel = torch.stack([t @ t.T for t in saved_kernel])
        p, q = torch.linalg.eigh(nystrom_kernel)
        p = p.to(X[0].device).float()
        torch.save(p, "p.pt")
        q = q.to(X[0].device).float()
        torch.save(q, "q.pt")
        eigenvalues_nystrom = p / X[0].shape[0]
        eigenfuncs_nystrom = lambda x: kernel(net, x, X, saved_kernel) @ q / p.unsqueeze(1) * math.sqrt(X[0].shape[0])
        end = time.time() - start
        return eigenvalues_nystrom, eigenfuncs_nystrom, end


    sigma = 1
    cov = (torch.eye(nystrom_samples, nystrom_samples) / sigma ** 2).to(device)
    start = time.time()
    count = 0
#     with torch.no_grad():

#         # We first take only nystrom_samples data to perform nystrom kernel calculation
#         # In other words, this loop runs only once
#         for x_batch, y_batch in train_loader:
#             x_batch = x_batch.float().to(device)
#             y_batch = y_batch.long().to(device)

#             # This is the subsampled data for the Monte Carlo calculation. We currently sample 100 of them
#             nys = x_batch[:nystrom_samples]
#             torch.save(nys, "initial_data.pt")

#             # We obtain the jacobian of subsampled data
#             jac_nystrom = torch.autograd.functional.jacobian(
#                 lambda *params: _stateless.functional_call(model, {n: p for n, p in zip(names, params)}, nys), tuple(model.parameters()))
#             _, eigfuns, _ = nystrom(model, jac_nystrom, ntk_kernel)
#             print("nystrom kernel calculation takes ", time.time() - start)
#             break

#         # Now we calculate all the eigenvalues of the whole training set
#         for x_batch, y_batch in train_loader:
#             x_batch = x_batch.float().to(device)
#             y_batch = y_batch.long().to(device)

#             jac_train = torch.autograd.functional.jacobian(
#                 lambda *params: _stateless.functional_call(model, {n: p for n, p in zip(names, params)}, x_batch), tuple(model.parameters()))
#             psi = eigfuns(jac_train).to(device)

#             # define loss function
#             def cross_ent(x):
#                 activation = torch.nn.LogSoftmax(dim=1)
#                 criteria = nn.CrossEntropyLoss()
#                 return criteria(activation(x), y_batch)

#             # Claim: this hessian is block diagonal, in the sense that we can just take the first element of each block
#             block_diag = torch.autograd.functional.hessian(cross_ent, model(x_batch))
#             hessian = torch.stack([block_diag[t][:, t, :].reshape(10, 10).to(device) for t in range(len(block_diag))])

#             batch_cov = torch.einsum('bik,kbj->bij', torch.einsum('ibj,bik->bjk', psi, hessian), psi)
#             cov += torch.sum(batch_cov, dim=0)
#             count += 1
#             print("batch " + str(count) + " done, " + str(250-count) + " remaining...")
#             print("time it took to do this batch: ", time.time() - start)

#     print(time.time() - start)
#     torch.save(cov, "cov.pt")
#     # Invert covariance matrix and complete training
#     cov = torch.linalg.solve(cov, torch.eye(nystrom_samples).to(device))
#     torch.save(cov, "cov.pt")
#     print("finish nystrom")

    # At this point everything that needs to be used to perform inference should have already been calculated.
    # That means if you terminate the program at this point, you should be able to recover the eigenfunctions
    # from the saved data and do whatever inference you like
    X = torch.load("load/initial_data.pt")
    jj = torch.autograd.functional.jacobian(lambda *params: _stateless.functional_call(model, {n: p for n, p in zip(names, params)}, X), tuple(model.parameters()))
    k0 = torch.load("load/ker.pt")
    q0 = torch.load("load/q.pt")
    p0 = torch.load("load/p.pt")
    eigfuns = lambda x: ntk_kernel(model, x, jj, k0) @ q0 / p0.unsqueeze(1) * math.sqrt(jj[0].shape[0])
    cov = torch.load("load/cov.pt")
    print(cov)


    yp = torch.zeros(10000,10)
    gt = torch.zeros(10000, dtype=torch.int32)
    import tensorflow_probability as tfp


    correct = 0
    total = batch_size
    prev = 0
    with torch.no_grad():
        for x_batch, y_batch in test_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            jac = torch.autograd.functional.jacobian(lambda *params: _stateless.functional_call(model, {n: p for n, p in zip(names, params)}, x_batch), tuple(model.parameters()))
            psi = eigfuns(jac).to(device)
            psi = torch.permute(psi, (1,0,2))

            std = torch.einsum('bik,bjk->bij', torch.stack([p @ cov for p in psi], dim=0), psi)
            m = torch.distributions.multivariate_normal.MultivariateNormal(model(x_batch), std)
            samples = []
            for _ in range(256):
                samples.append(m.sample())
            y_pred = torch.mean(torch.cat(samples, dim=0), dim=0).reshape(1, -1)

            yp[prev:total] = y_pred
            gt[prev:total] = y_batch

            _, predicted = torch.max(y_pred, 1)
            prev = total
            total += y_batch.size(0)
            correct += (predicted == y_batch).sum().item()

    print("ece score is", tfp.stats.expected_calibration_error(10, yp.cpu().numpy(), gt.cpu().numpy()))
    print('Accuracy of the Bayesian network on the 10000 test images: %d %%' % (100 * correct / prev))




    correct = 0
    total = batch_size
    prev = 0
    with torch.no_grad():
        for x_batch, y_batch in test_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            y_pred = model(x_batch)

            yp[prev:total] = y_pred
            gt[prev:total] = y_batch

            _, predicted = torch.max(y_pred, 1)
            prev = total
            total += y_batch.size(0)
            correct += (predicted == y_batch).sum().item()
    print("ece score is", tfp.stats.expected_calibration_error(10, yp.cpu().numpy(), gt.cpu().numpy()))
    print('Accuracy of the original network on the 10000 test images: %d %%' % (100 * correct / total))