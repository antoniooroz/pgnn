from typing import List

from pgnn.base import MCD_Base
from .gcn import GCN
class MCD_GCN(MCD_Base):  
    def __init__(self, nfeatures: int, nclasses: int, config, hiddenunits: List[int], drop_prob: float, adj_matrix):
        super().__init__(nfeatures, nclasses, config)

        self.model = GCN(nfeatures, nclasses, config, hiddenunits, drop_prob, adj_matrix)