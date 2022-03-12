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

# gt = {'0':1296, '1':48, '2':48, '3':48, '4':27648, '5':27648, '6':64, '7':64, '8':48, '9':48, '10':48, '11':48, '12':48, '13':48, '14':7680, '15':10}
# TODO list:
# 1. Is vjp twice faster than functional jacobian? (https://discuss.pytorch.org/t/computing-vector-jacobian-and-jacobian-vector-product-efficiently/69595/5)
# 2. We are currently linearizing the entire model. Can we linearize only the DEQ? That means the linear layer will not feature in our calculations and only function as a forward/backward pass



# This code is used later for calculating jacobian wrt model parameters
import contextlib
@contextlib.contextmanager
def reparametrize_module(module, parameters_and_buffers):
    # Parametrization does not support to change submodules directly
    for name, tensor in parameters_and_buffers.items():
        _apply_func_submodules(
            torch.nn.utils.parametrize.register_parametrization,
            module, name.split("."), (_ReparametrizedTensor(tensor),))
    yield
    for name in parameters_and_buffers:
        _apply_func_submodules(
            torch.nn.utils.parametrize.remove_parametrizations,
            module, name.split("."), (False,))


class _ReparametrizedTensor(torch.nn.Module):
    def __init__(self, tensor):
        super().__init__()
        self._tensor = tensor

    def forward(self, original):
        return self._tensor


def _apply_func_submodules(func, module, path, args):
    if len(path) == 1:
        func(module, path[0], *args)
    else:
        _apply_func_submodules(func, getattr(module, path[0]), path[1:], args)


def functional_call(module, parameters_and_buffers, args, kwargs=None):
    # TODO allow kwargs such as unsafe and others for parametrization
    if kwargs is None:
        kwargs = {}
    with reparametrize_module(module, parameters_and_buffers):
        if isinstance(args, tuple):
            out = module(*args, **kwargs)
        else:
            out = module(args, **kwargs)
    return out


# Not using the parser right now, just assuming gpu device
parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0)
args = parser.parse_args()

device = torch.device('cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

if __name__ == '__main__':

    model = ResDEQ(3, 48, 64, 10).float().to(device)
    model.load_state_dict(torch.load("model.pth"))
    model.eval()
    num_params = count_parameters(model)
    names = list(n for n, _ in model.named_parameters())
    
    # This is the nystrom parameter. Namely, how many data samples are we taking?
    # There are actually two hyperparameters in Nystrom: the number of Monte Carlo samples, and the number of top eigenvalues
    # Right now the assumption is that those two are the same, both taken to be 100
    k = 100


    # This is the kernel wrt which we calculated its nystrom
    def ntk_kernel(net, logit, jac1, jac2=None, saved_kernel=None):
        if jac2 == None:
            # if we are not doing inference, we simply want the jacobian
            # TODO: the code below should really just be wrapped into a function called collect_jacobian
            first_jacobian_element = jac1[0].reshape(k, 10, -1)[:, logit, :].reshape(k, 1, -1)
            idx = 0
            final_jac = torch.zeros(k, 1, 64842)
            final_jac[:, :, :first_jacobian_element.size()[2]] = first_jacobian_element
            idx += first_jacobian_element.size()[2]
            for i in range(1, len(jac1)):
                next_jacobian_element = jac1[i].reshape(k, 10, -1)[:, logit, :].reshape(k, 1, -1)
                final_jac[:, :, idx:idx + next_jacobian_element.size()[2]] = next_jacobian_element
                idx += next_jacobian_element.size()[2]
            # The code above is really just collecting the jacobian and putting them all into one vector
            return final_jac.reshape(k, -1)

        else:
            # Here we are doing inference using the nystrom kernel
            # input data is jac1, and we don't need jac2 because the info is in saved_kernel
            assert saved_kernel != None
            first_jacobian_element = jac1[0].reshape(10, -1)[logit, :].reshape(1, -1)
            idx = 0
            final_jac = torch.zeros(1, 64842)
            final_jac[:, :first_jacobian_element.size()[1]] = first_jacobian_element
            idx += first_jacobian_element.size()[1]
            for i in range(1, len(jac1)):
                next_jacobian_element = jac1[i].reshape(10, -1)[logit, :].reshape(1, -1)
                final_jac[:, idx:idx + next_jacobian_element.size()[1]] = next_jacobian_element
                idx += next_jacobian_element.size()[1]
            return (final_jac @ saved_kernel.T).to(device)

    # The only notable change in this function is that X is passed in not as raw data, but jacobian
    # This is so that we don't have to re-calculate the jacobian every time
    def nystrom(net, logit, X, k, kernel):
        start = time.time()
        saved_kernel = kernel(net, logit, X)
        torch.save(saved_kernel, "ker" + str(logit) + ".pt")
        K = saved_kernel @ saved_kernel.T
        # TODO: Change to torch
        p, q = eigh(K.data.cpu().numpy(), subset_by_index=[K.shape[0]-k, K.shape[0]-1])
        p = torch.from_numpy(p).to(X[0].device).float()[range(-1, -(k+1), -1)]
        torch.save(p, "p" + str(logit) + ".pt")
        q = torch.from_numpy(q).to(X[0].device).float()[:, range(-1, -(k+1), -1)]
        torch.save(q, "q" + str(logit) + ".pt")
        eigenvalues_nystrom = p / X[0].shape[0]
        eigenfuncs_nystrom = lambda x: kernel(net, logit, x, X, saved_kernel) @ q / p * math.sqrt(X[0].shape[0])
        end = time.time() - start
        return eigenvalues_nystrom, eigenfuncs_nystrom, end


    train_set, test_set, in_channels, num_classes = get_dataset('cifar10', True)
    train_loader = DataLoader(train_set, 50000, shuffle=True)
    test_loader = DataLoader(test_set, 1, shuffle=False)

    eigfun1 = None
    eigfun2 = None
    eigfun3 = None
    eigfun4 = None
    eigfun5 = None
    eigfun6 = None
    eigfun7 = None
    eigfun8 = None
    eigfun9 = None
    eigfun10 = None

    # TODO: find out the correct sigma, which represents the weight priors
    # Setting sigma to 1 is just me being lazy, although it does work experimentally
    sigma = 1
    cov = (torch.eye(k,k) / sigma ** 2).to(device)
    start = time.time()
    with torch.no_grad():
        for x_batch, y_batch in train_loader:
            x_batch = x_batch.float().to(device)
            y_batch = y_batch.long().to(device)
            
            # This is the subsampled data for the Monte Carlo calculation. We currently sample 100 of them
            nys = torch.zeros(k,3,32,32).to(device)
            for i in range(k):
                nys[i] = x_batch[i]
            torch.save(nys, "initial_data.pt")
            
            # We obtain the jacobian of subsampled data
            jac1 = torch.autograd.functional.jacobian(lambda *params: functional_call(model, {n: p for n, p in zip(names, params)}, nys), tuple(model.parameters()))
            eigval1, eigfun1, _ = nystrom(model, 0, jac1, k, ntk_kernel)
            eigval2, eigfun2, _ = nystrom(model, 1, jac1, k, ntk_kernel)
            eigval3, eigfun3, _ = nystrom(model, 2, jac1, k, ntk_kernel)
            eigval4, eigfun4, _ = nystrom(model, 3, jac1, k, ntk_kernel)
            eigval5, eigfun5, _ = nystrom(model, 4, jac1, k, ntk_kernel)
            eigval6, eigfun6, _ = nystrom(model, 5, jac1, k, ntk_kernel)
            eigval7, eigfun7, _ = nystrom(model, 6, jac1, k, ntk_kernel)
            eigval8, eigfun8, _ = nystrom(model, 7, jac1, k, ntk_kernel)
            eigval9, eigfun9, _ = nystrom(model, 8, jac1, k, ntk_kernel)
            eigval10, eigfun10, _ = nystrom(model, 9, jac1, k, ntk_kernel)
            print(time.time() - start)

            # Here we do "training" for nystrom
            for i in range(x_batch.size()[0]):
                psi = torch.zeros(10,k).to(device)
                jac2 = torch.autograd.functional.jacobian(lambda *params: functional_call(model, {n: p for n, p in zip(names, params)}, x_batch[i].reshape(1,3,32,32)), tuple(model.parameters()))
                psi[0] = eigfun1(jac2)
                psi[1] = eigfun2(jac2)
                psi[2] = eigfun3(jac2)
                psi[3] = eigfun4(jac2)
                psi[4] = eigfun5(jac2)
                psi[5] = eigfun6(jac2)
                psi[6] = eigfun7(jac2)
                psi[7] = eigfun8(jac2)
                psi[8] = eigfun9(jac2)
                psi[9] = eigfun10(jac2)

                # define loss function
                def cross_ent(x):
                    activation = torch.nn.LogSoftmax(dim=1)
                    criteria = nn.CrossEntropyLoss()
                    ans = criteria(activation(x), torch.unsqueeze(y_batch[i], 0))
                    return ans

                hessian = torch.autograd.functional.hessian(cross_ent, model(x_batch[i].reshape(1,3,32,32))).reshape(10,10).to(device)

                cov += psi.T @ hessian @ psi
    print(time.time() - start)
    torch.save(cov, "cov.pt")
    # Invert covariance matrix and complete training
    cov = torch.linalg.solve(cov, torch.eye(k).to(device))
    torch.save(cov, "cov.pt")
    print("finish nystrom")

    # At this point everything that needs to be used to perform inference should have already been calculated.
    # That means if you terminate the program at this point, you should be able to recover the eigenfunctions
    # from the saved data and do whatever inference you like


    correct = 0
    total = 0
    with torch.no_grad():
        for x_batch, y_batch in test_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            psi = torch.rand(10,k).to(device)
            jac = torch.autograd.functional.jacobian(lambda *params: functional_call(model, {n: p for n, p in zip(names, params)}, x_batch.reshape(1,3,32,32)), tuple(model.parameters()))
            psi[0] = eigfun1(jac)
            psi[1] = eigfun2(jac)
            psi[2] = eigfun3(jac)
            psi[3] = eigfun4(jac)
            psi[4] = eigfun5(jac)
            psi[5] = eigfun6(jac)
            psi[6] = eigfun7(jac)
            psi[7] = eigfun8(jac)
            psi[8] = eigfun9(jac)
            psi[9] = eigfun10(jac)

            m = torch.distributions.multivariate_normal.MultivariateNormal(model(x_batch), psi @ cov @ psi.T)
            result1 = m.sample()
            result2 = m.sample()
            result3 = m.sample()
            result4 = m.sample()
            result5 = m.sample()
            y_pred = torch.mean(torch.cat([result1, result2, result3, result4, result5], dim=0), dim=0).reshape(1, -1)

            _, predicted = torch.max(y_pred, 1)
            total += y_batch.size(0)
            correct += (predicted == y_batch).sum().item()
    print('Accuracy of the Bayesian network on the 10000 test images: %d %%' % (100 * correct / total))




    correct = 0
    total = 0
    with torch.no_grad():
        for x_batch, y_batch in test_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            y_pred = model(x_batch)
            _, predicted = torch.max(y_pred, 1)
            total += y_batch.size(0)
            correct += (predicted == y_batch).sum().item()
    print('Accuracy of the original network on the 10000 test images: %d %%' % (100 * correct / total))

