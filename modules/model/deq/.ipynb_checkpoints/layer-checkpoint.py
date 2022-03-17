import torch
from packaging import version
[...]
def _register_backward_hook(self, module):
    if version.parse(torch.__version__) >= version.parse("1.10"):
        module.register_full_backward_hook(self.backward_hook)
    else:
        module.register_backward_hook(self.backward_hook)

class DEQFixedPoint(torch.nn.Module):
    def __init__(self, f, solver, **kwargs):
        super().__init__()
        self.f = f
        self.solver = solver
        self.kwargs = kwargs

    def forward(self, x):
        # compute forward pass and re-engage autograd tape
        with torch.no_grad():
            z = torch.zeros_like(x)
            for _ in range(50):
                z = self.f(z, x)
#                 prev = z
#                 if torch.max(prev - z) > 0.1:
#                     z = self.f(z, x)
#             z, self.forward_res = self.solver(lambda z : self.f(z, x), torch.zeros_like(x), **self.kwargs)
        z = self.f(z, x)

#         if self.training:
#             z0 = z.clone().detach().requires_grad_()
#             f0 = self.f(z0,x)
#             def backward_hook(grad):
#                 g, self.backward_res = self.solver(lambda y : torch.autograd.grad(f0, z0, y, retain_graph=True)[0] + grad,
#                                                 grad, **self.kwargs)
#                 return g

#             z.register_hook(backward_hook)
        return z
