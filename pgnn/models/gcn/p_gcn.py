from typing import List

from .gcn import GCN

from pgnn.base import P_Base

class P_GCN(P_Base):
    def __init__(self, nfeatures: int, nclasses: int, config, hiddenunits: List[int], drop_prob: float, adj_matrix):
        super().__init__()
        self.config = config
        self.nclasses = nclasses
        self.nfeatures = nfeatures

        self.return_sites = ("obs", "final_logits", "_RETURN")

        self.model = GCN(
            nfeatures=nfeatures,
            nclasses=nclasses,
            config=config,
            hiddenunits=hiddenunits,
            drop_prob=drop_prob,
            adj_matrix=adj_matrix
            )

        self.pyronize(self.model)

    