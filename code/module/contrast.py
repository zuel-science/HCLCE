import torch
import torch.nn as nn


class Contrast(nn.Module):
    def __init__(self, hidden_dim, tau, lam):
        super(Contrast, self).__init__()
        self.proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.tau = tau
        self.lam = lam
        for model in self.proj:
            if isinstance(model, nn.Linear):
                nn.init.xavier_normal_(model.weight, gain=1.414)

    def sim(self, z1, z2):
        """
        计算两个张量之间的相似度（点积归一化）
        :param z1: tensor, [num_nodes, hidden_dim]
        :param z2: tensor, [num_nodes, hidden_dim]
        :return: 相似度矩阵
        """
        dot_numerator = torch.mm(z1, z2.t())  # 点积
        norm_z1 = torch.norm(z1, dim=-1, keepdim=True)
        norm_z2 = torch.norm(z2, dim=-1, keepdim=True)
        dot_denominator = torch.mm(norm_z1, norm_z2.t())
        sim_matrix = dot_numerator / (dot_denominator + 1e-8)
        return torch.exp(sim_matrix / self.tau)

    def forward(self, z_mp, z_ng, z_sm, pos):
        z_proj_mp = self.proj(z_mp)
        z_proj_ng = self.proj(z_ng)

        matrix_mp2ng = self.sim(z_proj_mp, z_proj_ng)
        matrix_ng2mp = matrix_mp2ng.t()
        
        matrix_mp2ng = matrix_mp2ng / (torch.sum(matrix_mp2ng, dim=1).view(-1, 1) + 1e-8)
        matrix_ng2mp = matrix_ng2mp / (torch.sum(matrix_ng2mp, dim=1).view(-1, 1) + 1e-8)

        lori_mg = -torch.log(matrix_mp2ng.mul(pos.to_dense()).sum(dim=-1)).mean()
        lori_np = -torch.log(matrix_ng2mp.mul(pos.to_dense()).sum(dim=-1)).mean()

        z_proj_sm = self.proj(z_sm)

        matrix_mp2sm = self.sim(z_proj_mp, z_proj_sm)
        matrix_sm2mp = matrix_mp2sm.t()

        matrix_mp2sm = matrix_mp2sm / (torch.sum(matrix_mp2sm, dim=1).view(-1, 1) + 1e-8)
        matrix_sm2mp = matrix_sm2mp / (torch.sum(matrix_sm2mp, dim=1).view(-1, 1) + 1e-8)

        lori_mm = -torch.log(matrix_mp2sm.mul(pos.to_dense()).sum(dim=-1)).mean()
        lori_sp = -torch.log(matrix_sm2mp.mul(pos.to_dense()).sum(dim=-1)).mean()

        return self.lam * (lori_mg + lori_np) + (1 - self.lam) * (lori_mm + lori_sp)
