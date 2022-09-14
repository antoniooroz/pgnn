from typing import List
import torch.nn as nn
from .ppnp import PPNP
from pgnn.base import P_Base

class P_PPNP(P_Base):
    def __init__(self, nfeatures: int, nclasses: int, config, hiddenunits: List[int], drop_prob: float, propagation: nn.Module):
        super().__init__()
        self.config = config
        self.nclasses = nclasses
        self.nfeatures = nfeatures

        self.return_sites = ("obs", "final_logits", "_RETURN")
        
        self.model = PPNP(
            nfeatures=nfeatures,
            nclasses=nclasses,
            config=config,
            hiddenunits=hiddenunits,
            drop_prob=drop_prob,
            propagation=propagation
            )

        self.pyronize(self.model)