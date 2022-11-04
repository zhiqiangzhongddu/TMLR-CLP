"""
Mainly Copy from C&S implementation
https://github.com/CUAI/CorrectAndSmooth
"""
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch_sparse import SparseTensor
from torch_geometric.utils import to_undirected

from copy import deepcopy
from scipy import sparse

import numpy as np
np.random.seed(0)


def sgc(x, adj, num_propagations):
    for _ in tqdm(range(num_propagations)):
        x = adj @ x
    return torch.from_numpy(x).to(torch.float)


def lp(adj, train_idx, labels, num_propagations, p, alpha, preprocess):
    if p is None:
        p = 0.6
    if alpha is None:
        alpha = 0.4
    
    c = labels.max() + 1
    idx = train_idx
    y = np.zeros((labels.shape[0], c))
    y[idx] = F.one_hot(labels[idx],c).numpy().squeeze(1)
    result = deepcopy(y)
    for i in tqdm(range(num_propagations)):
        result = y + alpha * adj @ (result**p)
        result = np.clip(result,0,1)
    return torch.from_numpy(result).to(torch.float)


def diffusion(x, adj, num_propagations, p, alpha):
    if p is None:
        p = 1.
    if alpha is None:
        alpha = 0.5

    inital_features = deepcopy(x)
    x = x **p
    for i in tqdm(range(num_propagations)):
        x = x - alpha * (sparse.eye(adj.shape[0]) - adj) @ x
        x = x **p
    return torch.from_numpy(x).to(torch.float)


def spectral(data, post_fix):
    from julia.api import Julia
    jl = Julia(compiled_modules=False)
    from julia import Main
    Main.include("./norm_spec.jl")
    print('Setting up spectral embedding')
    data.edge_index = to_undirected(data.edge_index)
    np_edge_index = np.array(data.edge_index.T)

    
    N = data.num_nodes
    row, col = data.edge_index
    adj = SparseTensor(row=row, col=col, sparse_sizes=(N, N))
    adj = adj.to_scipy(layout='csr')
    result = torch.tensor(Main.main(adj, 128)).float()
    torch.save(result, f'../input/embeddings/spectral{post_fix}.pt')
        
    return result


def preprocess(
        data, preprocess="diffusion", num_propagations=10, p=None, alpha=None, use_cache=True, post_fix=""
):
    if use_cache:
        try:
            x = torch.load(f'../input/embeddings/{preprocess}{post_fix}.pt')
            print('Using cache')
            return x
        except:
            print(f'../input/embeddings/{preprocess}{post_fix}.pt not found or not enough iterations! Regenerating it '
                  f'now')

    if preprocess == "spectral":
        return spectral(data, post_fix)
    
    print('Computing adj...')
    N = data.num_nodes
    data.edge_index = to_undirected(data.edge_index, data.num_nodes)

    row, col = data.edge_index
    adj = SparseTensor(row=row, col=col, sparse_sizes=(N, N))
    adj = adj.set_diag()
    deg = adj.sum(dim=1).to(torch.float)
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
    adj = deg_inv_sqrt.view(-1, 1) * adj * deg_inv_sqrt.view(1, -1)

    adj = adj.to_scipy(layout='csr')
        
    print(f'Start {preprocess} processing')

    if preprocess == "sgc":
        result = sgc(data.x.numpy(), adj, num_propagations)
    if preprocess == "diffusion":
        result = diffusion(data.x.numpy(), adj, num_propagations, p = p, alpha = alpha)

    torch.save(result, f'../input/embeddings/{preprocess}{post_fix}.pt')
    
    return result
    
