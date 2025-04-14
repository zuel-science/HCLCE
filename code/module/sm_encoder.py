import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

def dropout_feat(x, drop_prob):
    drop_mask = torch.empty(
        (x.size(1),),
        dtype=torch.float32,
        device=x.device).uniform_(0, 1) < drop_prob
    x = x.clone()
    x[:, drop_mask] = 0

    return x

def edge_drop(edge_index, edge_weight, drop_prob):
    # 生成随机保留的掩码
    num_edges = edge_index.size(1)
    mask = torch.rand(num_edges, device=edge_index.device) > drop_prob

    # 根据掩码更新 edge_index 和 edge_weight
    edge_index = edge_index[:, mask]
    edge_weight = edge_weight[mask]

    return edge_index, edge_weight

def construct_p_adj_from_mps(mps, num_p):
    # 初始化空的列表，用于存储所有矩阵的索引和权重
    all_indices = []
    all_values = []
    
    # 遍历 mps 中的所有稀疏矩阵
    for matrix in mps:
        matrix = matrix.coalesce()  # 确保稀疏矩阵是 coalesced
        all_indices.append(matrix.indices())  # 添加索引
        all_values.append(matrix.values())    # 添加权重
    
    # 将所有索引按列拼接
    pp_indices = torch.cat(all_indices, dim=1)
    # 将所有权重拼接
    pp_values = torch.cat(all_values)
    
    # 构造合并后的邻接矩阵并 coalesce
    pp = torch.sparse_coo_tensor(pp_indices, pp_values, size=(num_p, num_p)).coalesce()
    
    return pp

def sparse_to_edge_index(sparse_tensor):
    indices = sparse_tensor.indices()  # Shape: [2, nnz]
    values = sparse_tensor.values()    # Shape: [nnz]
    return indices, values

class GCNWithNoise(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers):
        super(GCNWithNoise, self).__init__()
        
        self.num_layers = num_layers
        
        # 第一层使用 input_dim，后续层使用 hidden_dim
        self.layers = nn.ModuleList()
        self.layers.append(GCNConv(input_dim, hidden_dim))  # 第一层
        for _ in range(num_layers - 1):
            self.layers.append(GCNConv(hidden_dim, hidden_dim))  # 后续层

    def forward(self, x, edge_index, edge_weight=None):
        for i in range(self.num_layers):
            x = self.layers[i](x, edge_index, edge_weight=edge_weight)
            x = F.relu(x)
        return x

def add_gaussian_noise(x, mean=0.0, std=0.1):
    noise = torch.randn_like(x) * std + mean
    return x + noise

class Sm_encoder(nn.Module):
    def __init__(self, hidden_dim, num_n, num_layers, pf, pe, noise_std=0.1):
        super(Sm_encoder, self).__init__()

        self.hidden_dim = hidden_dim
        self.num_n = num_n
        self.num_layers = num_layers
        self.pf = pf
        self.pe = pe
        self.noise_std = noise_std
        
        # 初始化 GCN
        self.gcn = GCNWithNoise(input_dim=hidden_dim, hidden_dim=hidden_dim, num_layers=num_layers)

    def forward(self, feat, mps):
        pp = construct_p_adj_from_mps(mps, self.num_n)
        edge_index, edge_weight = sparse_to_edge_index(pp)

        feat_dropped = dropout_feat(feat, self.pf)
        edge_index, edge_weight = edge_drop(edge_index, edge_weight, self.pe)

        feat_noisy = add_gaussian_noise(feat_dropped, std=self.noise_std)

        z_sm = self.gcn(feat_noisy, edge_index, edge_weight=edge_weight)

        return z_sm