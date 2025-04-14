import torch
import torch.nn as nn

class ContraNorm(nn.Module):
    def __init__(self, dim, scale=0.1, dual_norm=False, pre_norm=False, temp=1.0, learnable=False, positive=False, identity=False):
        super().__init__()
        if learnable and scale > 0:
            import math
            scale_init = math.log(scale) if positive else scale
            self.scale_param = nn.Parameter(torch.empty(dim).fill_(scale_init))
        self.dual_norm = dual_norm
        self.scale = scale
        self.pre_norm = pre_norm
        self.temp = temp
        self.learnable = learnable
        self.positive = positive
        self.identity = identity

        self.layernorm = nn.LayerNorm(dim, eps=1e-6)

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(0)
            added_batch_dim = True
        else:
            added_batch_dim = False

        if self.scale > 0.0:
            xn = nn.functional.normalize(x, dim=2)
            if self.pre_norm:
                x = xn

            sim = torch.bmm(xn, xn.transpose(1, 2)) / self.temp
            if self.dual_norm:
                sim = nn.functional.softmax(sim, dim=2) + nn.functional.softmax(sim, dim=1)
            else:
                sim = nn.functional.softmax(sim, dim=2)

            x_neg = torch.bmm(sim, x)

            if not self.learnable:
                if self.identity:
                    x = (1 + self.scale) * x - self.scale * x_neg
                else:
                    x = x - self.scale * x_neg
            else:
                scale = torch.exp(self.scale_param) if self.positive else self.scale_param
                scale = scale.view(1, 1, -1)
                if self.identity:
                    x = scale * x - scale * x_neg
                else:
                    x = x - scale * x_neg

        x = self.layernorm(x)

        if added_batch_dim:
            x = x.squeeze(0)

        return x