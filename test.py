import numpy as np
# np.set_printoptions(linewidth=20000, precision=4)

import torch
import torch.nn as nn

from functorch import vmap

def jacobian(model, x_i, vs):
    output = model(x_i)
    def vjp(v):
        return torch.autograd.grad(output, model.parameters(), v.view_as(output))
    result = vmap(vjp)(vs)
    return torch.cat([g.flatten(1) for g in result], 1).detach()

device = torch.device('cuda')
B = 128
N = 5
M = 5
model = nn.Sequential(
    nn.Linear(N, 32),
    nn.ReLU(),
    nn.Linear(32, 32),
    nn.ReLU(),
    nn.Linear(32, M),
).to(device)
x = torch.randn(B, N).to(device)
vs = torch.eye(M).to(x.device)

# compute the Jacobian sample-by-sample
model.eval()
print(jacobian(model, x[:1], vs).data.cpu().numpy())
print(x[:1].data.cpu().numpy())


print(jacobian(model, x[1:2], vs).data.cpu().numpy())
print(x[1:2].data.cpu().numpy())


print(jacobian(model, x[2:3], vs).data.cpu().numpy())
print(x[2:3].data.cpu().numpy())

# assert torch.allclose(jacobian(f, x), torch.diag(2 * x))
