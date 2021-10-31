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
#             print(x.size())
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
#         print(z.size())
#         print(x.size())
        y = F.relu(self.conv1(z))
        return F.relu(z + F.relu(x + self.conv2(y)))

    
class BBBLeNet(ModuleWrapper):
    '''The architecture of LeNet with Bayesian Layers'''

    def __init__(self, outputs, inputs, priors, layer_type='lrt', activation_type='softplus'):
        super(BBBLeNet, self).__init__()

        self.num_classes = outputs
        self.layer_type = layer_type
        self.priors = priors


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
        self.deq = DEQFixedPoint(ResNetLayer(120,1200, self.priors), broyden, threshold=100, eps=1) #eps cannot be any lower
#         self.deq = DEQFixedPoint(ResNetLayer(120,1200, self.priors), anderson, tol=1e-1, max_iter=100, beta=2.0)
        self.flatten = FlattenLayer(32 * 32 * 3)
        self.fc1 = BBBLinear(32 * 32 * 3, 120, bias=True, priors=self.priors)
        self.act = nn.ReLU()
        self.fc4 = BBBLinear(120, self.num_classes, bias=True, priors=self.priors)
        
#         return self.fc4(F.relu(self.fc1(self.flatten(x2)))), kl
from scipy.optimize import root
import time
from termcolor import colored


def _safe_norm(v):
    if not torch.isfinite(v).all():
        return np.inf
    return torch.norm(v)


def scalar_search_armijo(phi, phi0, derphi0, c1=1e-4, alpha0=1, amin=0):
    ite = 0
    phi_a0 = phi(alpha0)    # First do an update with step size 1
    if phi_a0 <= phi0 + c1*alpha0*derphi0:
        return alpha0, phi_a0, ite

    # Otherwise, compute the minimizer of a quadratic interpolant
    alpha1 = -(derphi0) * alpha0**2 / 2.0 / (phi_a0 - phi0 - derphi0 * alpha0)
    phi_a1 = phi(alpha1)

    # Otherwise loop with cubic interpolation until we find an alpha which
    # satisfies the first Wolfe condition (since we are backtracking, we will
    # assume that the value of alpha is not too small and satisfies the second
    # condition.
    while alpha1 > amin:       # we are assuming alpha>0 is a descent direction
        factor = alpha0**2 * alpha1**2 * (alpha1-alpha0)
        a = alpha0**2 * (phi_a1 - phi0 - derphi0*alpha1) - \
            alpha1**2 * (phi_a0 - phi0 - derphi0*alpha0)
        a = a / factor
        b = -alpha0**3 * (phi_a1 - phi0 - derphi0*alpha1) + \
            alpha1**3 * (phi_a0 - phi0 - derphi0*alpha0)
        b = b / factor

        alpha2 = (-b + torch.sqrt(torch.abs(b**2 - 3 * a * derphi0))) / (3.0*a)
        phi_a2 = phi(alpha2)
        ite += 1

        if (phi_a2 <= phi0 + c1*alpha2*derphi0):
            return alpha2, phi_a2, ite

        if (alpha1 - alpha2) > alpha1 / 2.0 or (1 - alpha2/alpha1) < 0.96:
            alpha2 = alpha1 / 2.0

        alpha0 = alpha1
        alpha1 = alpha2
        phi_a0 = phi_a1
        phi_a1 = phi_a2

    # Failed to find a suitable step length
    return None, phi_a1, ite


def line_search(update, x0, g0, g, nstep=0, on=True):
    """
    `update` is the propsoed direction of update.
    Code adapted from scipy.
    """
    tmp_s = [0]
    tmp_g0 = [g0]
    tmp_phi = [torch.norm(g0)**2]
    s_norm = torch.norm(x0) / torch.norm(update)

    def phi(s, store=True):
        if s == tmp_s[0]:
            return tmp_phi[0]    # If the step size is so small... just return something
        x_est = x0 + s * update
        g0_new = g(x_est)
        phi_new = _safe_norm(g0_new)**2
        if store:
            tmp_s[0] = s
            tmp_g0[0] = g0_new
            tmp_phi[0] = phi_new
        return phi_new
    
    if on:
        s, phi1, ite = scalar_search_armijo(phi, tmp_phi[0], -tmp_phi[0], amin=1e-2)
    if (not on) or s is None:
        s = 1.0
        ite = 0

    x_est = x0 + s * update
    if s == tmp_s[0]:
        g0_new = tmp_g0[0]
    else:
        g0_new = g(x_est)
    return x_est, g0_new, x_est - x0, g0_new - g0, ite

def rmatvec(part_Us, part_VTs, x):
    # Compute x^T(-I + UV^T)
    # x: (N, 2d, L')
    # part_Us: (N, 2d, L', threshold)
    # part_VTs: (N, threshold, 2d, L')
    if part_Us.nelement() == 0:
        return -x
    xTU = torch.einsum('bij, bijd -> bd', x, part_Us)   # (N, threshold)
    return -x + torch.einsum('bd, bdij -> bij', xTU, part_VTs)    # (N, 2d, L'), but should really be (N, 1, (2d*L'))


def matvec(part_Us, part_VTs, x):
    # Compute (-I + UV^T)x
    # x: (N, 2d, L')
    # part_Us: (N, 2d, L', threshold)
    # part_VTs: (N, threshold, 2d, L')
    if part_Us.nelement() == 0:
        return -x
    VTx = torch.einsum('bdij, bij -> bd', part_VTs, x)  # (N, threshold)
    return -x + torch.einsum('bijd, bd -> bij', part_Us, VTx)     # (N, 2d, L'), but should really be (N, (2d*L'), 1)


def broyden(f, x0, threshold, eps=1e-3, stop_mode="rel", ls=False, name="unknown"):
    bsz, total_hsize, H, W = x0.size()
    seq_len = H * W
    g = lambda y: f(y) - y
    dev = x0.device
    alternative_mode = 'rel' if stop_mode == 'abs' else 'abs'
    
    x_est = x0           # (bsz, 2d, L')
    gx = g(x_est)        # (bsz, 2d, L')
    nstep = 0
    tnstep = 0
    
    # For fast calculation of inv_jacobian (approximately)
    Us = torch.zeros(bsz, total_hsize, seq_len, threshold).to(dev)     # One can also use an L-BFGS scheme to further reduce memory
    VTs = torch.zeros(bsz, threshold, total_hsize, seq_len).to(dev)
    update = -matvec(Us[:,:,:,:nstep], VTs[:,:nstep], gx)      # Formally should be -torch.matmul(inv_jacobian (-I), gx)
    prot_break = False
    
    # To be used in protective breaks
    protect_thres = (1e6 if stop_mode == "abs" else 1e3) * seq_len
    new_objective = 1e8

    trace_dict = {'abs': [],
                  'rel': []}
    lowest_dict = {'abs': 1e8,
                   'rel': 1e8}
    lowest_step_dict = {'abs': 0,
                        'rel': 0}
    nstep, lowest_xest, lowest_gx = 0, x_est, gx

    while nstep < threshold:
        x_est, gx, delta_x, delta_gx, ite = line_search(update, x_est, gx, g, nstep=nstep, on=ls)
        nstep += 1
        tnstep += (ite+1)
        abs_diff = torch.norm(gx).item()
        rel_diff = abs_diff / (torch.norm(gx + x_est).item() + 1e-9)
        diff_dict = {'abs': abs_diff,
                     'rel': rel_diff}
        trace_dict['abs'].append(abs_diff)
        trace_dict['rel'].append(rel_diff)
        for mode in ['rel', 'abs']:
            if diff_dict[mode] < lowest_dict[mode]:
                if mode == stop_mode: 
                    lowest_xest, lowest_gx = x_est.clone().detach(), gx.clone().detach()
                lowest_dict[mode] = diff_dict[mode]
                lowest_step_dict[mode] = nstep

        new_objective = diff_dict[stop_mode]
        if new_objective < eps: break
        if new_objective < 3*eps and nstep > 30 and np.max(trace_dict[stop_mode][-30:]) / np.min(trace_dict[stop_mode][-30:]) < 1.3:
            # if there's hardly been any progress in the last 30 steps
            break
        if new_objective > trace_dict[stop_mode][0] * protect_thres:
            prot_break = True
            break

        part_Us, part_VTs = Us[:,:,:,:nstep-1], VTs[:,:nstep-1]
        vT = rmatvec(part_Us, part_VTs, delta_x)
        u = (delta_x - matvec(part_Us, part_VTs, delta_gx)) / torch.einsum('bij, bij -> b', vT, delta_gx)[:,None,None]
        vT[vT != vT] = 0
        u[u != u] = 0
        VTs[:,nstep-1] = vT
        Us[:,:,:,nstep-1] = u
        update = -matvec(Us[:,:,:,:nstep], VTs[:,:nstep], gx)

    # Fill everything up to the threshold length
    for _ in range(threshold+1-len(trace_dict[stop_mode])):
        trace_dict[stop_mode].append(lowest_dict[stop_mode])
        trace_dict[alternative_mode].append(lowest_dict[alternative_mode])

    return {"result": lowest_xest,
            "lowest": lowest_dict[stop_mode],
            "nstep": lowest_step_dict[stop_mode],
            "prot_break": prot_break,
            "abs_trace": trace_dict['abs'],
            "rel_trace": trace_dict['rel'],
            "eps": eps,
            "threshold": threshold}
     
        
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
        
    def forward(self, x, **kwargs):
        z0 = torch.zeros_like(x)

        # Forward pass
        with torch.no_grad():
            z_star = self.solver(lambda z: self.f(z, x), z0, **self.kwargs)['result']
            new_z_star = z_star
#         print(new_z_star)
        # (Prepare for) Backward pass, see step 3 above
#         if self.training:
        new_z_star = self.f(z_star.requires_grad_(), x)

        # Jacobian-related computations, see additional step above. For instance:
#         jac_loss = jac_loss_estimate(new_z_star, z_star, vecs=1)

        def backward_hook(grad):
            if self.hook is not None:
                self.hook.remove()
                torch.cuda.synchronize()   # To avoid infinite recursion
            # Compute the fixed point of yJ + grad, where J=J_f is the Jacobian of f at z_star
            new_grad = self.solver(lambda y: autograd.grad(new_z_star, z_star, y, retain_graph=True)[0] + grad, \
                                   torch.zeros_like(grad), **self.kwargs)['result']
            return new_grad

        self.hook = new_z_star.register_hook(backward_hook)
        return new_z_star
        
#     def forward(self, x):
#         # compute forward pass and re-engage autograd tape
#         with torch.no_grad():
#             z = self.solver(lambda z : self.f(z, x), torch.zeros_like(x), **self.kwargs)['result']
#         z = self.f(z,x)
        
#         # set up Jacobian vector product (without additional forward calls)
#         z0 = z.clone().detach().requires_grad_()
#         f0 = self.f(z0,x)
#         def backward_hook(grad):
#             g = self.solver(lambda y : autograd.grad(f0, z0, y, retain_graph=True)[0] + grad,
#                                                grad, **self.kwargs)['result']
#             return g
                
#         z.register_hook(backward_hook)
#         return z