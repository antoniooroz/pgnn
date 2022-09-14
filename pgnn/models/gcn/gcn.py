##############################################################
# This file is a modified version from the following source
# Author: Thomas N. Kipf and Max Welling
# Last Visited: 14.06.2022
# Title: Graph Convolutional Networks
# URL: https://github.com/tkipf/gcn
##############################################################

from typing import List
import torch
import torch.nn as nn

from pgnn.utils import edge_dropout, get_device, preprocess_adj
from pgnn.base import GraphConvolution, Base

class GCN(Base):
    def __init__(self, nfeatures: int, nclasses: int, config, hiddenunits: List[int], drop_prob: float, adj_matrix):
        super().__init__()
        self.config = config
        self.nclasses = nclasses
        self.nfeatures = nfeatures

        self.original_adj_matrix = adj_matrix.toarray()
        self.preprocessed_adj_matrix = torch.tensor(preprocess_adj(adj_matrix, laplacian=True).toarray()).type(torch.float32).to(get_device())
        self.isolated_adj_matrix = torch.eye(self.original_adj_matrix.shape[0]).type(torch.float32).to(get_device())

        hiddenunits = [nfeatures] + hiddenunits + [nclasses]

        layers = []
        for i in range(len(hiddenunits)-1):            
            layers.append(GraphConvolution(
                input_dim=hiddenunits[i], 
                output_dim=hiddenunits[i+1],
                config=config,  
                bias=self.config.bias,
                activation=nn.ReLU() if i < len(hiddenunits)-2 else lambda x: x,
                dropout=drop_prob if (i>0 or not self.config.disable_dropout_on_input) else 0,
                name="layer"+str(i)
                ))

        self.layers = nn.Sequential(*layers)

    def forward(self, attr_matrix: torch.sparse.FloatTensor, idx: torch.LongTensor):
        if self.config.network_effects and not (self.config.train_without_network_effects and self.training):
            support = self.adj_matrix_with_edge_dropout()
        else:
            support = self.isolated_adj_matrix

        input = (attr_matrix, support)

        local_logits, _ = self.layers(input)
        final_logits = local_logits.index_select(dim=-2, index=idx)

        return final_logits

    def adj_matrix_with_edge_dropout(self):
        if self.config.edge_drop_prob > 0 and self.config.mode == "DE-GCN":
            return torch.tensor(preprocess_adj(edge_dropout(self.original_adj_matrix, self.config.edge_drop_prob), laplacian=True).toarray()).type(torch.float32).to(get_device())
        else:
            return self.preprocessed_adj_matrix