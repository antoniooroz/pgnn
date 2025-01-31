##############################################################
# This file is a modified version from the following source
# Author: Maximilian Stadler, Bertrand Charpentier, Simon Geisler, Daniel Zügner and Stephan Günnemann
# Last Visited: 14.06.2022
# Title: Graph Posterior Network
# URL: https://github.com/stadlmax/Graph-Posterior-Network
##############################################################
from typing import Callable, Optional
from torch_geometric.typing import Adj, OptTensor

import torch
from torch import Tensor
import torch.nn.functional as F
from torch_sparse import SparseTensor, matmul
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.conv.gcn_conv import gcn_norm


class LabelPropagation(MessagePassing):
    """ simple label propagation baseline

    code taken from the torch_geometric repository on GitHub (https://github.com/rusty1s/pytorch_geometric)
    """

    def __init__(self, num_layers: int, alpha: float):
        super(LabelPropagation, self).__init__(aggr='add')
        self.num_layers = num_layers
        self.alpha = alpha

    @torch.no_grad()
    def forward(
            self, y: Tensor, edge_index: Adj, mask: Optional[Tensor] = None,
            edge_weight: OptTensor = None,
            post_step: Callable = lambda y: y.clamp_(0., 1.)
        ) -> Tensor:

        if y.dtype == torch.long:
            y = F.one_hot(y.view(-1)).to(torch.float)

        out = y
        if mask is not None:
            out = torch.zeros_like(y)
            out[mask] = y[mask]

        if isinstance(edge_index, SparseTensor) and not edge_index.has_value():
            edge_index = gcn_norm(edge_index, add_self_loops=False)
        elif isinstance(edge_index, Tensor) and edge_weight is None:
            edge_index, edge_weight = gcn_norm(edge_index, num_nodes=y.size(0),
                                               add_self_loops=False)

        res = (1 - self.alpha) * out
        for _ in range(self.num_layers):
            # propagate_type: (y: Tensor, edge_weight: OptTensor)
            out = self.propagate(edge_index, x=out, edge_weight=edge_weight,
                                 size=None)
            out.mul_(self.alpha).add_(res)
            out = post_step(out)

        return out


    def message(self, x_j: Tensor, edge_weight: OptTensor) -> Tensor:
        return x_j if edge_weight is None else edge_weight.view(-1, 1) * x_j

    def message_and_aggregate(self, adj_t: SparseTensor, x: Tensor) -> Tensor:
        return matmul(adj_t, x, reduce=self.aggr)

    def __repr__(self):
        return '{}(num_layers={}, alpha={})'.format(self.__class__.__name__,
                                                    self.num_layers,
                                                    self.alpha)
