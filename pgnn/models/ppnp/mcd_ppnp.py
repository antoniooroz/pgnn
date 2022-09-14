from typing import List
import torch.nn as nn

from pgnn.base import MCD_Base
from .ppnp import PPNP

class MCD_PPNP(MCD_Base):  
    def __init__(self, nfeatures: int, nclasses: int, config, hiddenunits: List[int], drop_prob: float, propagation: nn.Module):
        super().__init__(nfeatures, nclasses, config)

        self.model = PPNP(nfeatures, nclasses, config, hiddenunits, drop_prob, propagation)