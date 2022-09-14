import torch
import torch.nn as nn
import pyro
from pyro.nn import PyroModule, PyroParam, PyroSample
from pgnn.base import P_Base
from pgnn.utils import get_device, get_edge_indices, get_self_edge_indices
import pyro.distributions as dist
from .gat import GAT

class P_GAT(P_Base):

    def __init__(self, nfeatures, nclasses, adj_matrix, config):
        super().__init__()
        self.nfeatures = nfeatures
        self.nclasses = nclasses 
        self.config = config
        self.return_sites = ("obs", "final_logits", "_RETURN", "edge_indices")

        if self.config.mode not in ["P-PROJ-GAT", "Mixed-PROJ-GAT"]:
            self.return_sites += tuple(["edge_scores"])


        self.model = GAT(
            nfeatures=nfeatures,
            nclasses=nclasses,
            config=config,
            adj_matrix=adj_matrix
            )

        self.pyronize(self.model)

    def forward(self, attr_matrix, idx, y=None):
        final_logits = self.model(attr_matrix, idx)

        pyro.deterministic("final_logits", final_logits)

        edge_indices = self.model.edge_indices if self.config.network_effects and not (self.config.train_without_network_effects and self.training) else self.model.self_edge_indices
        
        pyro.deterministic("edge_indices", edge_indices)
        if "edge_scores" in self.return_sites: 
            pyro.deterministic("edge_scores", torch.cat([layer.edge_scores for layer in filter(lambda x: x.scoring_fn_target_mean is not None, self.model.layers)], dim=-1).to(edge_indices.device))

        # Sampling
        with pyro.plate("data", idx.shape[0]):
            obs = pyro.sample("obs", dist.Categorical(logits=final_logits), obs=y)

        return final_logits