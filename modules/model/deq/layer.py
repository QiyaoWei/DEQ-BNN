import torch

class DEQFixedPoint(torch.nn.Module):
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
        if self.training:
            z0 = z.clone().detach().requires_grad_()
            f0 = self.f(z0,x)
            def backward_hook(grad):
                g, self.backward_res = self.solver(lambda y : torch.autograd.grad(f0, z0, y, retain_graph=True)[0] + grad,
                                                grad, **self.kwargs)
                return g
                    
            z.register_hook(backward_hook)
        return z
        