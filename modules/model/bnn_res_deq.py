from torch import nn
from torch.nn import functional as F
from blitz.modules import BayesianConv2d, BayesianLinear, BayesianModule
from blitz.utils import variational_estimator
from .deq import DEQFixedPoint, anderson

class BayesianResBasicBlock(nn.Module):
    def __init__(self, n_channels, n_inner_channels, kernel_size=3, num_groups=8):
        super().__init__()

        self.conv1 = BayesianConv2d(n_channels, n_inner_channels, (kernel_size,kernel_size), padding=kernel_size//2, bias=False)
        self.conv2 = BayesianConv2d(n_inner_channels, n_channels, (kernel_size,kernel_size), padding=kernel_size//2, bias=False)
        self.norm1 = nn.GroupNorm(num_groups, n_inner_channels)
        self.norm2 = nn.GroupNorm(num_groups, n_channels)
        self.norm3 = nn.GroupNorm(num_groups, n_channels)
        
    def forward(self, z, x):
        y = self.norm1(F.relu(self.conv1(z)))
        return self.norm3(F.relu(z + self.norm2(x + self.conv2(y))))

@variational_estimator
class BNNResDEQ(nn.Module):
    def __init__(self, in_channels, deq_channels, inner_channels, num_classes, tol=1e-2, max_iter=50) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.conv1 = BayesianConv2d(in_channels, deq_channels, (3,3), padding=1)
        self.bn1 = nn.BatchNorm2d(deq_channels)
        self.deq = DEQFixedPoint(BayesianResBasicBlock(deq_channels, inner_channels), anderson, tol=tol, max_iter=max_iter)
        self.bn2 = nn.BatchNorm2d(deq_channels)
        self.ap = nn.AvgPool2d(8,8)
        self.flatten = nn.Flatten()
        self.linear = BayesianLinear(deq_channels*4*4, num_classes)

    def forward(self, x):
        for m in self.modules():
            if isinstance(m, BayesianModule):
                m.resample()
        
        for m in self.children():
            x = m(x)
        return x