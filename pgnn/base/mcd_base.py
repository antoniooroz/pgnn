from pgnn.base import Base
from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import Base

import pgnn.base.uncertainty_estimation as UE


class MCD_Base(Base):
    
    def __init__(self, nfeatures, nclasses, config):
        super().__init__()
        self.nfeatures = nfeatures
        self.nclasses = nclasses
        self.config = config
        self.model = None
    
    def forward(self, attr_matrix: torch.sparse.FloatTensor, idx: torch.LongTensor):
        return self.model.forward(attr_matrix, idx)

    def get_predictions(self, attr_matrix, idx):
        idx = idx.to(attr_matrix.device)
        
        logits = []
        for _ in range(self.config.prediction_samples_num):
            final_logits = self.forward(attr_matrix, idx)
            logits.append(final_logits.unsqueeze(0))
        
        logits = torch.cat(logits, dim=0).to(idx.device)
        
        if self.config.pred_score=="softmax": 
            probs_all = F.softmax(logits, dim=-1)
            probs_mean = (torch.sum(probs_all, dim=0) / self.config.prediction_samples_num).squeeze(0)    
            preds = probs_mean.max(-1).indices
            epist = UE.get_uncertainty(self.config, self.config.uncertainty, probs_all=probs_all, probs_mean=probs_mean, preds=preds, logits=logits)

            alea = UE.get_uncertainty(self.config, "probability", probs_all=probs_all, probs_mean=probs_mean, preds=preds, logits=logits)
        else: 
            raise NotImplementedError()
        
        return probs_mean, preds, epist, alea
    
    def load_model(self, name, attr_mat):
        self.model.load_model(name[name.index("-")+1:], attr_mat)
        
    def log_weights(self):
        return self.model.log_weights()

    def eval(self):
        """
        Adapted from https://discuss.pytorch.org/t/using-dropout-in-evaluation-mode/27721
        Last visited: 14.06.2022
        """
        super().eval()
        if self.config.mode.startswith('MCD-'):
            for m in self.model.modules():
                if m.__class__.__name__.startswith('Dropout') or m.__class__.__name__.startswith('MixedDropout'):
                    m.train()