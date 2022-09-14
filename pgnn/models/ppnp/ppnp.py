##############################################################
# This file is a modified version from the following source
# Author: Johannes Gasteiger, Aleksandar Bojchevski and Stephan Günnemann
# Last Visited: 14.06.2022
# Title: PPNP and APPNP
# URL: https://github.com/gasteigerjo/ppnp
##############################################################

from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F

from pgnn.base import Linear, Base

import copy
import os

class PPNP(Base):    
    def __init__(self, nfeatures: int, nclasses: int, config, hiddenunits: List[int], drop_prob: float, propagation: nn.Module):
        super().__init__()
        self.config = config
        self.nclasses = nclasses
        self.nfeatures = nfeatures

        hiddenunits = [nfeatures] + hiddenunits + [nclasses]
        
        layers = []
        for i in range(len(hiddenunits)-1):
            layers.append(Linear(
                input_dim=hiddenunits[i], 
                output_dim=hiddenunits[i+1],
                config=config,  
                bias=self.config.bias,
                activation=nn.ReLU() if i < len(hiddenunits)-2 else lambda x: x,
                dropout=drop_prob if (i>0 or not self.config.disable_dropout_on_input) else 0,
                name="layer"+str(i)
                ))

        self.layers = nn.Sequential(*layers)

        self.propagation = propagation

    def forward(self, attr_matrix: torch.sparse.FloatTensor, idx: torch.LongTensor):
        local_logits = self.layers(attr_matrix)
        
        if self.config.network_effects and not (self.config.train_without_network_effects and self.training):
            final_logits = self.propagation(local_logits, attr_matrix, idx) # With network effects
        else:
            final_logits = local_logits.index_select(dim=-2, index=idx) # Without network effects

        return final_logits