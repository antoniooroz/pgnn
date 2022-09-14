##############################################################
# Parts of this file contain content from the following source
# Author: Johannes Gasteiger, Aleksandar Bojchevski and Stephan Günnemann
# Last Visited: 14.06.2022
# Title: PPNP and APPNP
# URL: https://github.com/gasteigerjo/ppnp
##############################################################

from pyro.infer.predictive import Predictive
import torch
import torch.nn.functional as F

from pgnn.utils import get_statistics, config_to_dict, get_device, escape_slashes
import pgnn.base.uncertainty_estimation as UE

import pyro
from pyro.nn import PyroModule
from pyro.infer import SVI, TraceMeanField_ELBO, Predictive
from pyro.infer.autoguide import AutoNormal

from pyro import poutine

import pyro.distributions as dist
import os

import copy

class P_Base(PyroModule):
    def __init__(self):
        super().__init__()
        
        self.config = None
        self.nclasses = None
        self.nfeatures = None
        self.fcs = None
        self.dropout = None
        self.act_fn = None
        self.propagation = None
        self.return_sites = None

    def forward(self, attr_matrix: torch.sparse.FloatTensor, idx: torch.LongTensor, y = None):
        final_logits = self.model(attr_matrix, idx)
        pyro.deterministic("final_logits", final_logits)

        # Sampling
        with pyro.plate("data", idx.shape[0]):
            obs = pyro.sample("obs", dist.Categorical(logits=final_logits), obs=y)

        return final_logits

    def get_predictions(self, attr_matrix, idx):
        idx = idx.to(attr_matrix.device)
        predictive = Predictive(self, guide=self.guide, num_samples=self.config.prediction_samples_num,
            return_sites=self.return_sites, parallel=True)
        result = predictive(attr_matrix, idx)
        
        if self.config.pred_score=="softmax": 
            logits = result['final_logits']
            probs_all = F.softmax(logits, dim=-1)
            probs_mean = (torch.sum(probs_all, dim=0) / self.config.prediction_samples_num).squeeze(0)    
            preds = probs_mean.max(-1).indices
            epist = UE.get_uncertainty(self.config, self.config.uncertainty, probs_all=probs_all, probs_mean=probs_mean, preds=preds, logits=logits, result=result, idx=idx)

            alea = UE.get_uncertainty(self.config, "entropy", probs_all=probs_all, probs_mean=probs_mean, preds=preds, logits=logits, result=result, idx=idx)
        elif self.config.pred_score=="distribution":
            obs = result['obs'].T
            classes = torch.tensor(list(range(self.nclasses))).reshape(-1,1,1).repeat(1,obs.shape[0], obs.shape[1]).to(obs.device)
            occurences = torch.zeros(classes.shape).to(obs.device)
            occurences[classes == obs] = 1
            counts = occurences.sum(dim=-1).T
            probs_mean = counts / counts.sum(dim=1).repeat(counts.shape[1],1).T
            preds = probs_mean.max(-1).indices
            epist = UE.get_uncertainty(self.config, self.config.uncertainty, probs_mean=probs_mean, preds=preds, result=result, idx=idx)
            alea = UE.get_uncertainty(self.config, "entropy", probs_mean=probs_mean, preds=preds, result=result, idx=idx)
        else: 
            raise NotImplementedError()
        
        return probs_mean, preds, epist, alea
    
    def training_init(self, pytorch_seed, model_name, iteration, data_seed):
        torch.manual_seed(seed=pytorch_seed)
        self.set_optimizers_and_svi()
        self.pytorch_seed = pytorch_seed
        self.model_name = model_name
        self.iteration = iteration
        self.data_seed = data_seed
        
    def get_loss(self, attr_mat_norm, idx, labels, oods):
        self.set_eval(True)
        idx=idx[oods==0]
        labels=labels[oods==0]
        with poutine.scale(scale=1.0/idx.shape[0]):
            loss = self.svis[self.era].evaluate_loss(attr_mat_norm, idx, labels)  
        return loss

    def training_step(self, phase, attr_mat_norm, idx, labels, oods):
        ########################################################################
        # Get Loss                                                             #
        # Take Step if Training Phase                                          #
        ########################################################################
        id_idx=idx[oods==0]
        id_labels=labels[oods==0]
        with poutine.scale(scale=1.0/idx.shape[0]):
            if phase == 'train':
                self.set_eval(False)
                loss = self.svis[self.era].step(attr_mat_norm, id_idx, id_labels)
            else:
                self.set_eval(True)
                # SVI Loss
                loss = self.svis[self.era].evaluate_loss(attr_mat_norm, id_idx, id_labels)  

        ########################################################################
        # Get Accuracies                                                       #
        ########################################################################    
        torch.set_grad_enabled(False)
        self.guide.requires_grad_(False)
        probs, preds, confidence_all, confidence_correct, confidence_false, datapoints_correct, datapoints_false = get_statistics(self, attr_mat_norm, idx, labels, oods)
        
        return loss, probs, confidence_all, confidence_correct, confidence_false, datapoints_correct, datapoints_false

    def custom_state_dict(self, acc):
        return {
            'model': copy.deepcopy(self.state_dict()),
            'guide': copy.deepcopy(self.guide.state_dict()),
            'pyro_params': copy.deepcopy(pyro.get_param_store().get_state()),
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
        
        # Fixes loading error
        idx = torch.tensor([0]).to(get_device())
        labels = torch.tensor([0]).to(get_device())     
        self.set_eval(False)
        self.svis[self.era].step(attr_mat, idx, labels)

        pyro.clear_param_store()
        self.guide.load_state_dict(state_dict['guide'])
        pyro.get_param_store().set_state(state_dict['pyro_params'])
        
        self.load_state_dict(state_dict['model'])        
        
    def load_model(self, name, attr_mat):
        state_dict = torch.load(escape_slashes(os.getcwd() + '/saved_models/' + name, map_location=get_device()))
        self.load_model_from_state_dict(state_dict, attr_mat)
        
    def set_eval(self, eval = True):
        if eval:
            self.eval()
            torch.set_grad_enabled(False)
        else:
            self.train()
            torch.set_grad_enabled(True)

        self.guide.requires_grad_(not eval)
            
    def log_weights(self):
        return self.model.log_weights()
    
    def set_optimizers_and_svi(self):
        self.optimizers = []
        self.svis = []
        
        self.guide = AutoNormal(self, init_scale=self.config.guide_init_scale)
        self.elbo = TraceMeanField_ELBO(num_particles=self.config.training_samples_num, vectorize_particles=self.config.vectorize)
        
        for era in range(self.config.number_of_eras):
            if self.config.optimizers[era] == "training":
                optimizer_class = pyro.optim.Adam if self.config.optim=="adam" else pyro.optim.ClippedAdam
                optimizer = optimizer_class({
                    "lr": self.config.learning_rate[era], 
                    "weight_decay": self.config.reg_lambda[era],
                    "clip_norm": self.config.clip_norm[era],
                    "lrd": self.config.lr_decay[era]
                })
                svi = SVI(self, self.guide, optimizer, loss=self.elbo)
            else:
                raise NotImplementedError()
            
            self.optimizers.append(optimizer)
            self.svis.append(svi)
    
    def set_era(self, era):
        self.era = era

    def pyronize(self, model):
        num_layers = len(model.layers)
        for i, layer in enumerate(model.layers): 
            if (self.config.mode[:self.config.mode.find("-")] == "Mixed") and i < num_layers - 1:
                continue

            layer.pyronize_weights()