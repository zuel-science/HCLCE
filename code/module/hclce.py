import torch.nn as nn
import torch.nn.functional as F
from .mp_encoder import Mp_encoder
from .ng_encoder import Ng_encoder
from .sm_encoder import Sm_encoder
from .contrast import Contrast
from .contra_norm import ContraNorm

class HCLCE(nn.Module):
    def __init__(self, hidden_dim, feats_dim_list, feat_drop, attn_drop, P, sample_rate,
                 nei_num, tau, lam, num_n, num_layers, pf, pe):
        super(HCLCE, self).__init__()
        self.hidden_dim = hidden_dim
        self.fc_list = nn.ModuleList([nn.Linear(feats_dim, hidden_dim, bias=True)
                                      for feats_dim in feats_dim_list])
        for fc in self.fc_list:
            nn.init.xavier_normal_(fc.weight, gain=1.414)

        if feat_drop > 0:
            self.feat_drop = nn.Dropout(feat_drop)
        else:
            self.feat_drop = lambda x: x
        self.mp = Mp_encoder(P, hidden_dim, attn_drop)
        self.sc = Ng_encoder(hidden_dim, sample_rate, nei_num, attn_drop)
        self.sp = Sm_encoder(hidden_dim, num_n, num_layers, pf, pe)
        self.contrast = Contrast(hidden_dim, tau, lam)
        self.contranorm = ContraNorm(dim=64, scale=0.1, dual_norm=False, pre_norm=False, temp=0.8, learnable=False, positive=False, identity=False)

    def forward(self, feats, pos, mps, nei_index):
        h_all = []
        for i in range(len(feats)):
            h_all.append(F.elu(self.feat_drop(self.fc_list[i](feats[i]))))
        z_mp = self.mp(h_all[0], mps)
        z_ng = self.sc(h_all, nei_index)
        z_sm = self.sp(h_all[0], mps)

        z_ng = self.contranorm(z_ng)
        
        loss = self.contrast(z_mp, z_ng, z_sm, pos)
        return loss

    def get_embeds(self, feats, mps):
        z_mp = F.elu(self.fc_list[0](feats[0]))
        z_mp = self.mp(z_mp, mps)
        return z_mp.detach()
