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
    num_edges = edge_index.size(1)
    mask = torch.rand(num_edges, device=edge_index.device) > drop_prob

    edge_index = edge_index[:, mask]
    edge_weight = edge_weight[mask]

    return edge_index, edge_weight

def construct_p_adj_from_mps(mps, num_p):
    all_indices = []
    all_values = []
    
    for matrix in mps:
        matrix = matrix.coalesce()
        all_indices.append(matrix.indices())
        all_values.append(matrix.values())
    
    pp_indices = torch.cat(all_indices, dim=1)
    pp_values = torch.cat(all_values)
    
    pp = torch.sparse_coo_tensor(pp_indices, pp_values, size=(num_p, num_p)).coalesce()
    
    return pp

def sparse_to_edge_index(sparse_tensor):
    indices = sparse_tensor.indices()
    values = sparse_tensor.values()
    return indices, values

class GCNWithNoise(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers):
        super(GCNWithNoise, self).__init__()
        
        self.num_layers = num_layers
        
        self.layers = nn.ModuleList()
        self.layers.append(GCNConv(input_dim, hidden_dim))
        for _ in range(num_layers - 1):
            self.layers.append(GCNConv(hidden_dim, hidden_dim))

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
        
        self.gcn = GCNWithNoise(input_dim=hidden_dim, hidden_dim=hidden_dim, num_layers=num_layers)

    def forward(self, feat, mps):
        pp = construct_p_adj_from_mps(mps, self.num_n)
        edge_index, edge_weight = sparse_to_edge_index(pp)

        feat_dropped = dropout_feat(feat, self.pf)
        edge_index, edge_weight = edge_drop(edge_index, edge_weight, self.pe)

        feat_noisy = add_gaussian_noise(feat_dropped, std=self.noise_std)

        z_sm = self.gcn(feat_noisy, edge_index, edge_weight=edge_weight)

        return z_sm
