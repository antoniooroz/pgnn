##############################################################
# This file is a modified version from the following source
# Author: Johannes Gasteiger, Aleksandar Bojchevski and Stephan Günnemann
# Last Visited: 14.06.2022
# Title: PPNP and APPNP
# URL: https://github.com/gasteigerjo/ppnp
##############################################################

import numpy as np
import scipy.sparse as sp
import torch
from torch.utils.data import TensorDataset, DataLoader
import torch_geometric.utils as tu
import wandb
import os

def preprocess_adj(adj, laplacian=True, self_loop=True):
    """Preprocessing of adjacency matrix for simple GCN model and conversion to tuple representation."""
    if self_loop:
        adj = adj + sp.eye(adj.shape[0])

    if laplacian:
        adj_normalized = normalize_adj(adj)
    else:
        adj_normalized = sp.coo_matrix(adj / adj.sum(axis=1))

    return adj_normalized

def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()

def sparse_matrix_to_torch(X):
    coo = X.tocoo()
    indices = np.array([coo.row, coo.col])
    return torch.sparse.FloatTensor(
            torch.LongTensor(indices),
            torch.FloatTensor(coo.data),
            coo.shape)

def matrix_to_torch(X):
    if sp.issparse(X):
        return torch.FloatTensor(X.todense()).to(get_device())
    else:
        return torch.FloatTensor(X)

def edge_dropout(adj, p):
    adj = torch.tensor(adj)
    dropout = torch.zeros(adj.shape)
    dropout = dropout.bernoulli(p=p)
    adj[dropout==1] = 0
    return adj.numpy()

def get_statistics(model, attr_mat_norm, idx, labels, oods, use_cached = False):
    probs, preds, epist, alea = model.get_predictions(attr_mat_norm, idx)
                    
    probs_max = torch.max(probs, dim=1)

    id_epist = epist[oods==0]
    id_preds = preds[oods==0]
    id_labels = labels[oods==0]
    
    # Confidence for all data points
    confidence_all = id_epist.sum()
    
    # Confidences for only correctly/ wrongly predicted data points
    confidence_correct = id_epist[id_preds==id_labels].sum()
    confidence_false = id_epist[id_preds!=id_labels].sum()

    # Number correctly/wrongly predicted data points
    datapoints_correct = (id_preds==id_labels).sum()
    datapoints_false = (id_preds!=id_labels).sum()
    
    return probs, preds, confidence_all, confidence_correct, confidence_false, datapoints_correct, datapoints_false

def final_run(model, attr_matrix, idx, labels, oods, batch_size=None):
    if batch_size is None:
        batch_size = idx.numel()
    dataset = TensorDataset(idx)
    dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False)
    
    running_confidence_all = 0
    running_confidence_correct = 0
    running_confidence_false = 0
    running_datapoints_correct = 0
    running_datapoints_false = 0
    running_loss = 0
    running_datapoints = 0
    
    labels = torch.tensor(labels).to(attr_matrix.device)
    
    for idx, in dataloader:        
        idx = idx.to(attr_matrix.device)
        _, _, confidence_all, confidence_correct, confidence_false, datapoints_correct, datapoints_false = get_statistics(model, attr_matrix, idx, labels[idx], oods[idx])
        running_confidence_all += confidence_all
        running_confidence_correct += confidence_correct
        running_confidence_false += confidence_false
        running_datapoints_correct += datapoints_correct
        running_datapoints_false += datapoints_false
        datapoints = datapoints_correct + datapoints_false
        running_datapoints += datapoints
        running_loss += model.get_loss(attr_matrix, idx, labels[idx], oods[idx]) * datapoints
        
    log_acc = (running_datapoints_correct / running_datapoints).item()
    log_conf_correct = (running_confidence_correct / running_datapoints_correct).item()
    log_conf_false = (running_confidence_false / running_datapoints_false).item()
    log_conf_all = (running_confidence_all / running_datapoints).item()
    log_loss = (running_loss / running_datapoints).item()

    return {
        "log_acc": log_acc, 
        "log_conf_correct": log_conf_correct, 
        "log_conf_false": log_conf_false, 
        "log_conf_all": log_conf_all,
        "log_loss": log_loss
    }
    
def config_to_dict(config):
    dictionary = {}
    for key in config.keys():
        dictionary[key] = config[key]
        
    return dictionary

def config_from_dict(dictionary):
    config = wandb.config
    for key in dictionary.keys():
        config[key] = dictionary[key]
        
    return config

def get_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif torch.backends.mps.is_available():
        return torch.device('mps')
    else:
        return torch.device('cpu')

def get_edge_indices(adj):      
    # Adj normally expected to be csr_tensor
    if not torch.is_tensor(adj):  
        adj = torch.tensor(adj.todense())
    edge_indices, edge_weights = tu.dense_to_sparse(adj.cpu())
        
    return edge_indices.to(get_device()), edge_weights.to(get_device())

def get_self_edge_indices(adj_csr_matrix):
    adj = torch.eye(adj_csr_matrix.shape[0])
    edge_indices, edge_weights = tu.dense_to_sparse(adj)
        
    return edge_indices.to(get_device())
    
def escape_slashes(val):
    if os.name == 'nt':
        return val.replace('/','\\')
    else:
        return  val