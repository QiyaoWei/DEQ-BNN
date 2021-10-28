import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
from metric import calculate_kl as KL_DIV
import torch.autograd as autograd


class ModuleWrapper(nn.Module):
    """Wrapper for nn.Module with support for arbitrary flags and a universal forward pass"""

    def __init__(self):
        super(ModuleWrapper, self).__init__()

    def set_flag(self, flag_name, value):
        setattr(self, flag_name, value)
        for m in self.children():
            if hasattr(m, 'set_flag'):
                m.set_flag(flag_name, value)

    def forward(self, x):
        for module in self.children():
#             print(module)
            x = module(x)

        kl = 0.0
        for module in self.modules():
            if hasattr(module, 'kl_loss'):
                kl = kl + module.kl_loss()

        return x, kl


class FlattenLayer(ModuleWrapper):

    def __init__(self, num_features):
        super(FlattenLayer, self).__init__()
        self.num_features = num_features

    def forward(self, x):
        return x.view(-1, self.num_features)
    

class BBBConv2d(ModuleWrapper):
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, bias=True, priors=None):

        super(BBBConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = 1
        self.use_bias = bias
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        if priors is None:
            priors = {
                'prior_mu': 0,
                'prior_sigma': 0.1,
                'posterior_mu_initial': (0, 0.1),
                'posterior_rho_initial': (-3, 0.1),
            }
        self.prior_mu = priors['prior_mu']
        self.prior_sigma = priors['prior_sigma']
        self.posterior_mu_initial = priors['posterior_mu_initial']
        self.posterior_rho_initial = priors['posterior_rho_initial']

        self.W_mu = Parameter(torch.empty((out_channels, in_channels, *self.kernel_size), device=self.device))
        self.W_rho = Parameter(torch.empty((out_channels, in_channels, *self.kernel_size), device=self.device))
        self.W_sigma = torch.log1p(torch.exp(self.W_rho))

        if self.use_bias:
            self.bias_mu = Parameter(torch.empty((out_channels), device=self.device))
            self.bias_rho = Parameter(torch.empty((out_channels), device=self.device))
            self.bias_sigma = torch.log1p(torch.exp(self.bias_rho))
        else:
            self.register_parameter('bias_mu', None)
            self.register_parameter('bias_rho', None)

        self.reset_parameters()

    def reset_parameters(self):
        self.W_mu.data.normal_(*self.posterior_mu_initial)
        self.W_rho.data.normal_(*self.posterior_rho_initial)

        if self.use_bias:
            self.bias_mu.data.normal_(*self.posterior_mu_initial)
            self.bias_rho.data.normal_(*self.posterior_rho_initial)

    def forward(self, input, sample=True):
        if self.training or sample:
            W_eps = torch.empty(self.W_mu.size()).normal_(0, 1).to(self.device)
            self.W_sigma = torch.log1p(torch.exp(self.W_rho))
            weight = self.W_mu + W_eps * self.W_sigma

            if self.use_bias:
                bias_eps = torch.empty(self.bias_mu.size()).normal_(0, 1).to(self.device)
                self.bias_sigma = torch.log1p(torch.exp(self.bias_rho))
                bias = self.bias_mu + bias_eps * self.bias_sigma
            else:
                bias = None
        else:
            weight = self.W_mu
            bias = self.bias_mu if self.use_bias else None

        return F.conv2d(input, weight, bias, self.stride, self.padding, self.dilation, self.groups)

    def kl_loss(self):
        kl = KL_DIV(self.prior_mu, self.prior_sigma, self.W_mu, self.W_sigma)
        if self.use_bias:
            kl += KL_DIV(self.prior_mu, self.prior_sigma, self.bias_mu, self.bias_sigma)
        return kl
    
    
class BBBLinear(ModuleWrapper):
    def __init__(self, in_features, out_features, bias=True, priors=None):
        super(BBBLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.use_bias = bias
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        if priors is None:
            priors = {
                'prior_mu': 0,
                'prior_sigma': 0.1,
                'posterior_mu_initial': (0, 0.1),
                'posterior_rho_initial': (-3, 0.1),
            }
        self.prior_mu = priors['prior_mu']
        self.prior_sigma = priors['prior_sigma']
        self.posterior_mu_initial = priors['posterior_mu_initial']
        self.posterior_rho_initial = priors['posterior_rho_initial']

        self.W_mu = Parameter(torch.empty((out_features, in_features), device=self.device))
        self.W_rho = Parameter(torch.empty((out_features, in_features), device=self.device))
        self.W_sigma = torch.log1p(torch.exp(self.W_rho))

        if self.use_bias:
            self.bias_mu = Parameter(torch.empty((out_features), device=self.device))
            self.bias_rho = Parameter(torch.empty((out_features), device=self.device))
            self.bias_sigma = torch.log1p(torch.exp(self.bias_rho))
        else:
            self.register_parameter('bias_mu', None)
            self.register_parameter('bias_rho', None)

        self.reset_parameters()

    def reset_parameters(self):
        self.W_mu.data.normal_(*self.posterior_mu_initial)
        self.W_rho.data.normal_(*self.posterior_rho_initial)

        if self.use_bias:
            self.bias_mu.data.normal_(*self.posterior_mu_initial)
            self.bias_rho.data.normal_(*self.posterior_rho_initial)

    def forward(self, input, sample=True):
        if self.training or sample:
            W_eps = torch.empty(self.W_mu.size()).normal_(0, 1).to(self.device)
            self.W_sigma = torch.log1p(torch.exp(self.W_rho))
            weight = self.W_mu + W_eps * self.W_sigma

            if self.use_bias:
                bias_eps = torch.empty(self.bias_mu.size()).normal_(0, 1).to(self.device)
                self.bias_sigma = torch.log1p(torch.exp(self.bias_rho))
                bias = self.bias_mu + bias_eps * self.bias_sigma
            else:
                bias = None
        else:
            weight = self.W_mu
            bias = self.bias_mu if self.use_bias else None

        return F.linear(input, weight, bias)

    def kl_loss(self):
        kl = KL_DIV(self.prior_mu, self.prior_sigma, self.W_mu, self.W_sigma)
        if self.use_bias:
            kl += KL_DIV(self.prior_mu, self.prior_sigma, self.bias_mu, self.bias_sigma)
        return kl
    
    
class ResNetLayer(ModuleWrapper):
    def __init__(self, n_channels, n_inner_channels, priors):
        super(ResNetLayer, self).__init__()
#         self.fc2 = BBBLinear(n_channels, n_inner_channels, bias=True, priors=priors)
#         self.fc3 = BBBLinear(n_inner_channels, n_channels, bias=True, priors=priors)
        self.conv1 = BBBConv2d(3, 9, 3, padding=3//2, bias=True, priors=priors)
        self.conv2 = BBBConv2d(9, 3, 5, padding=5//2, bias=True, priors=priors)
        
    def forward(self, z, x):
#         x2 = F.relu(self.fc2(x1))
#         x3 = F.relu(self.fc3(x2))
#         print("x3 = ", x3.size())
#         print("x = ", x.size())
#         print("z = ", z.size())
        y = F.relu(self.conv1(z))
        return F.relu(z + self.conv2(F.relu(self.conv1(x) + y)))

    
class BBBLeNet(ModuleWrapper):
    '''The architecture of LeNet with Bayesian Layers'''

    def __init__(self, outputs, inputs, priors, layer_type='lrt', activation_type='softplus'):
        super(BBBLeNet, self).__init__()

        self.num_classes = outputs
        self.layer_type = layer_type
        self.priors = priors
        self.outputs = outputs


#         if layer_type=='lrt':
#             BBBLinear = BBB_LRT_Linear
#             BBBConv2d = BBB_LRT_Conv2d
#         elif layer_type=='bbb':
#             BBBLinear = BBBLinear
#             BBBConv2d = BBBConv2d
#         else:
#             raise ValueError("Undefined layer_type")
        
#         if activation_type=='softplus':
#             self.act = nn.Softplus
#         elif activation_type=='relu':
#             self.act = nn.ReLU
#         else:
#             raise ValueError("Only softplus or relu supported")

#         self.conv1 = BBBConv2d(inputs, 6, 5, padding=0, bias=True, priors=self.priors)
#         self.act1 = self.act()
#         self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

#         self.conv2 = BBBConv2d(6, 16, 5, padding=0, bias=True, priors=self.priors)
#         self.act2 = self.act()
#         self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

#         self.flatten = FlattenLayer(5 * 5 * 16)
#         self.fc1 = BBBLinear(5 * 5 * 16, 120, bias=True, priors=self.priors)
#         self.act3 = self.act()

#         self.fc2 = BBBLinear(120, 84, bias=True, priors=self.priors)
#         self.act4 = self.act()

#         self.fc3 = BBBLinear(84, outputs, bias=True, priors=self.priors)
        
#     def forward(self, x):
#         self.conv1 = BBBConv2d(3, 9, 3, padding=kernel_size//2, bias=True, priors=self.priors)
#         self.conv2 = BBBConv2d(9, 3, 5, padding=kernel_size//2, bias=True, priors=self.priors)
#         self.norm1 = nn.GroupNorm(8, 9)
#         self.norm2 = nn.GroupNorm(8, 3)
#         self.norm = nn.GroupNorm(8, 3)
#         self.conv1.weight.data.normal_(0, 0.01)
#         self.conv2.weight.data.normal_(0, 0.01)
        self.deq = DEQFixedPoint(ResNetLayer(120,1200, self.priors), anderson, tol=1e-1, max_iter=100, beta=2.0)
        self.flatten = FlattenLayer(32 * 32 * 3)
        self.fc1 = BBBLinear(32 * 32 * 3, 120, bias=True, priors=self.priors)
        self.act = nn.ReLU()
        self.fc4 = BBBLinear(120, self.outputs, bias=True, priors=self.priors)
        
#         return self.fc4(F.relu(self.fc1(self.flatten(x2)))), kl
        
        
def anderson(f, x0, m=5, lam=1e-4, max_iter=50, tol=1e-2, beta = 1.0):
    """ Anderson acceleration for fixed point iteration. """
    bsz, d, H, W = x0.shape
    X = torch.zeros(bsz, m, d*H*W, dtype=x0.dtype, device=x0.device)
    F = torch.zeros(bsz, m, d*H*W, dtype=x0.dtype, device=x0.device)
    X[:,0], F[:,0] = x0.view(bsz, -1), f(x0).view(bsz, -1)
    X[:,1], F[:,1] = F[:,0], f(F[:,0].view_as(x0)).view(bsz, -1)
    
    H = torch.zeros(bsz, m+1, m+1, dtype=x0.dtype, device=x0.device)
    H[:,0,1:] = H[:,1:,0] = 1
    y = torch.zeros(bsz, m+1, 1, dtype=x0.dtype, device=x0.device)
    y[:,0] = 1
    
    res = []
    for k in range(2, max_iter):
        n = min(k, m)
        G = F[:,:n]-X[:,:n]
        H[:,1:n+1,1:n+1] = torch.bmm(G,G.transpose(1,2)) + lam*torch.eye(n, dtype=x0.dtype,device=x0.device)[None]
        alpha = torch.solve(y[:,:n+1], H[:,:n+1,:n+1])[0][:, 1:n+1, 0]   # (bsz x n)
        
        X[:,k%m] = beta * (alpha[:,None] @ F[:,:n])[:,0] + (1-beta)*(alpha[:,None] @ X[:,:n])[:,0]
        F[:,k%m] = f(X[:,k%m].view_as(x0)).view(bsz, -1)
        res.append((F[:,k%m] - X[:,k%m]).norm().item()/(1e-5 + F[:,k%m].norm().item()))
        if (res[-1] < tol):
            break
    return X[:,k%m].view_as(x0), res




class DEQFixedPoint(nn.Module):
    def __init__(self, f, solver, **kwargs):
        super().__init__()
        self.f = f
        self.solver = solver
        self.kwargs = kwargs
        
    def forward(self, x):
        # compute forward pass and re-engage autograd tape
        with torch.no_grad():
            z, self.forward_res = self.solver(lambda z : self.f(z, x), torch.zeros_like(x), **self.kwargs)
        z = self.f(z,x)
        
        # set up Jacobian vector product (without additional forward calls)
        z0 = z.clone().detach().requires_grad_()
        f0 = self.f(z0,x)
        def backward_hook(grad):
            g, self.backward_res = self.solver(lambda y : autograd.grad(f0, z0, y, retain_graph=True)[0] + grad,
                                               grad, **self.kwargs)
            return g
                
        z.register_hook(backward_hook)
        return z