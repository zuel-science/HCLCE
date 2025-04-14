import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class fus_att(nn.Module):
    def __init__(self, hidden_dim, attn_drop):
        super(fus_att, self).__init__()
        self.fc = nn.Linear(hidden_dim, hidden_dim, bias=True)
        nn.init.xavier_normal_(self.fc.weight, gain=1.414)

        self.tanh = nn.Tanh()
        self.att = nn.Parameter(torch.empty(size=(1, hidden_dim)), requires_grad=True)
        nn.init.xavier_normal_(self.att.data, gain=1.414)

        self.softmax = nn.Softmax(dim=-1)
        
        if attn_drop:
            self.attn_drop = nn.Dropout(attn_drop)
        else:
            self.attn_drop = lambda x: x

    def forward(self, embeds):
        beta = []
        attn_curr = self.attn_drop(self.att)
        for embed in embeds:

            sp = self.tanh(self.fc(embed))
            beta.append(torch.matmul(sp, attn_curr.T).mean(dim=0))
            
        beta = torch.cat(beta, dim=-1).view(-1)
        beta = self.softmax(beta)
        z_mc = 0
        
        for i in range(len(embeds)):
            z_mc += embeds[i] * beta[i]
            
        return z_mc


class nei_att(nn.Module):
    def __init__(self, hidden_dim, attn_drop):
        super(nei_att, self).__init__()
        self.att = nn.Parameter(torch.empty(size=(1, 2*hidden_dim)), requires_grad=True)
        nn.init.xavier_normal_(self.att.data, gain=1.414)
        if attn_drop:
            self.attn_drop = nn.Dropout(attn_drop)
        else:
            self.attn_drop = lambda x: x

        self.softmax = nn.Softmax(dim=1)
        self.leakyrelu = nn.LeakyReLU()

    def forward(self, nei, h, h_refer):
        nei_emb = F.embedding(nei, h)
        h_refer = torch.unsqueeze(h_refer, 1)
        h_refer = h_refer.expand_as(nei_emb)
        all_emb = torch.cat([h_refer, nei_emb], dim=-1)
        attn_curr = self.attn_drop(self.att)
        att = self.leakyrelu(all_emb.matmul(attn_curr.t()))
        att = self.softmax(att)
        nei_emb = (att*nei_emb).sum(dim=1)
        return nei_emb


class GlobalEncoder(nn.Module):
    def __init__(self, hidden_dim, attn_drop):
        super(GlobalEncoder, self).__init__()

        self.att = nn.Parameter(torch.empty(size=(1, hidden_dim)), requires_grad=True)
        nn.init.xavier_normal_(self.att.data, gain=1.414)

        self.attn_drop = nn.Dropout(attn_drop) if attn_drop else lambda x: x
        self.softmax = nn.Softmax(dim=0)

    def forward(self, features_list):
        all_nodes = torch.cat(features_list, dim=0)
        attn_curr = self.attn_drop(self.att)
        attn_scores = torch.matmul(all_nodes, attn_curr.T)  # (num_all_nodes, 1)
        attn_scores = attn_scores / np.sqrt(features_list[0].size(1))
        attn_scores = self.softmax(attn_scores)
        global_embed = (attn_scores * all_nodes).sum(dim=0)  # (hidden_dim, )

        return global_embed


class WeightedGate(nn.Module):
    def __init__(self, initial_gate_value=0.8):
        super(WeightedGate, self).__init__()
        self.gate = nn.Parameter(torch.tensor(initial_gate_value))
        self.sigmoid = nn.Sigmoid()

    def forward(self, z_mc, global_embed):
        gate_value = self.sigmoid(self.gate)  
        # 融合局部和全局信息，给z_mc更大的权重
        z_fused = gate_value * z_mc + (1 - gate_value) * global_embed
        return z_fused


class Ng_encoder(nn.Module):
    def __init__(self, hidden_dim, sample_rate, nei_num, attn_drop):
        super(Ng_encoder, self).__init__()
        self.intra = nn.ModuleList([nei_att(hidden_dim, attn_drop) for _ in range(nei_num)])
        self.inter = fus_att(hidden_dim, attn_drop)
        self.sample_rate = sample_rate
        self.nei_num = nei_num
        
        self.global_encoder = GlobalEncoder(hidden_dim, attn_drop)
        self.simplified_gate = WeightedGate()

    def forward(self, nei_h, nei_index):
        embeds = []
        for i in range(self.nei_num):
            sele_nei = []
            sample_num = self.sample_rate[i]
            for per_node_nei in nei_index[i]:
                if len(per_node_nei) >= sample_num:
                    select_one = torch.tensor(np.random.choice(per_node_nei, sample_num, replace=False))[np.newaxis]
                else:
                    select_one = torch.tensor(np.random.choice(per_node_nei, sample_num, replace=True))[np.newaxis]
                sele_nei.append(select_one)
            sele_nei = torch.cat(sele_nei, dim=0).cuda()
            one_type_emb = F.elu(self.intra[i](sele_nei, nei_h[i + 1], nei_h[0]))
            embeds.append(one_type_emb)
        z_mc = self.inter(embeds)

        global_embed = self.global_encoder(nei_h)

        z_fused = self.simplified_gate(z_mc, global_embed)

        return z_fused