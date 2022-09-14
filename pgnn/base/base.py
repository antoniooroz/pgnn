##############################################################
# Parts of this file contain content from the following source
# Author: Johannes Gasteiger, Aleksandar Bojchevski and Stephan Günnemann
# Last Visited: 14.06.2022
# Title: PPNP and APPNP
# URL: https://github.com/gasteigerjo/ppnp
##############################################################

from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F

from pgnn.utils import get_device, get_statistics, config_to_dict, escape_slashes
import pgnn.base.uncertainty_estimation as UE
from pyro.nn import PyroModule

import copy
import os

class Base(PyroModule):
    def __init__(self):
        super().__init__()
        self.nfeatures = None
        self.nclasses = None
        self.config = None
        self.fcs = None
        self.reg_params = None
        self.dropout = None
        self.act_fn = None
    
    def forward(self, attr_matrix: torch.sparse.FloatTensor, idx: torch.LongTensor):
        raise NotImplementedError()
    
    def get_predictions(self, attr_matrix, idx):
        idx = idx.to(attr_matrix.device)
        final_logits = self.forward(attr_matrix, idx)
        
        probs = F.softmax(final_logits, dim=-1)
        
        probs_max = probs.max(-1)
        preds = probs_max.indices
        alea = probs_max.values

        epist = UE.entropy({"probs_mean": probs})
        
        return probs, preds, epist, alea
    
    def training_init(self, pytorch_seed, model_name, iteration, data_seed):
        torch.manual_seed(seed=pytorch_seed)
        self.set_optimizers_and_reg_lambda()
        self.pytorch_seed = pytorch_seed
        self.model_name = model_name
        self.iteration = iteration
        self.data_seed = data_seed
        
    def get_loss(self, attr_mat_norm, idx, labels, oods):
        self.set_eval(True)
        probs, preds, confidence_all, confidence_correct, confidence_false, datapoints_correct, datapoints_false = get_statistics(self, attr_mat_norm, idx, labels, oods)

        probs_for_loss = probs[oods == 0]
        labels_for_loss = labels[oods == 0]
        # Calculate loss
        loss = F.nll_loss(torch.log(probs_for_loss), labels_for_loss)
        
        return loss.item()
    
    def training_step(self, phase, attr_mat_norm, idx, labels, oods):
        self.optimizers[self.era].zero_grad()
        self.set_eval(phase != 'train')
        with torch.set_grad_enabled(phase == 'train'):
            probs, preds, confidence_all, confidence_correct, confidence_false, datapoints_correct, datapoints_false = get_statistics(self, attr_mat_norm, idx, labels, oods)

            probs_for_loss = probs[oods==0]
            labels_for_loss = labels[oods==0]

            # Calculate loss
            loss = F.nll_loss(torch.log(probs_for_loss), labels_for_loss)

            if phase == 'train':
                loss.backward()
                self.optimizers[self.era].step()
                
        return loss.item(), probs, confidence_all, confidence_correct, confidence_false, datapoints_correct, datapoints_false
    
    def custom_state_dict(self, acc):
        return {
            'model': copy.deepcopy(self.state_dict()),
            'config': copy.deepcopy(config_to_dict(self.config)),
            'stopping_acc': copy.deepcopy(acc),
            'torch_seed': copy.deepcopy(self.pytorch_seed),
            'data_seed': copy.deepcopy(self.data_seed),
            'iteration': copy.deepcopy(self.iteration),
            'model_name': copy.deepcopy(self.model_name)
        }
           
    def save_model(self, custom_state_dict = None):
        if custom_state_dict is None:
            custom_state_dict = self.custom_state_dict(None)
        torch.save(custom_state_dict, escape_slashes(os.getcwd() + '/saved_models/' + self.config.mode + ' [' + self.model_name + '] [' + str(self.data_seed) + '] [' + str(self.iteration) + ']' + '.save'))
        
    def load_model_from_state_dict(self, state_dict, attr_mat):
        self.training_init(state_dict["torch_seed"], state_dict["model_name"], state_dict["iteration"], state_dict["data_seed"])
        self.load_state_dict(state_dict['model'])
        
    def load_model(self, name, attr_mat):
        state_dict = torch.load(escape_slashes(os.getcwd() + '/saved_models/' + name), map_location=get_device())
        self.load_model_from_state_dict(state_dict, attr_mat)
        
    def set_eval(self, eval = True):
        if eval:
            self.eval()
            torch.set_grad_enabled(False)
        else:
            self.train()
            torch.set_grad_enabled(True)
            
    def log_weights(self):
        log = {}
        for layer in self.layers:
            logWeights = layer.log_weights()
            for logWeight in logWeights:
                log[logWeight.name] = logWeight
        return log

    def set_optimizers_and_reg_lambda(self):
        self.optimizers = []
        self.reg_lambdas = []
        
        for era in range(self.config.number_of_eras):
            if self.config.optimizers[era] == "training":
                optimizer = torch.optim.Adam(self.parameters(), lr=self.config.learning_rate[era], weight_decay=self.config.reg_lambda[era])
            else:
                raise NotImplementedError()
            
            self.optimizers.append(optimizer)
    
    def set_era(self, era):
        self.era = era