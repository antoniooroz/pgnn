from pgnn.base import MCD_Base
from pgnn.models import GAT

class MCD_GAT(MCD_Base):  
    def __init__(self, nfeatures, nclasses, adj_matrix, config):
        super().__init__(nfeatures, nclasses, config)

        self.model = GAT(nfeatures, nclasses, adj_matrix, config)