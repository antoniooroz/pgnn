##############################################################
# This file is a modified version from the following source
# Author: Maximilian Stadler, Bertrand Charpentier, Simon Geisler, Daniel Zügner and Stephan Günnemann
# Last Visited: 14.06.2022
# Title: Graph Posterior Network
# URL: https://github.com/stadlmax/Graph-Posterior-Network
##############################################################

import datetime
from typing import Dict, Tuple, List
from .gpn_fns.utils.config import ModelConfiguration
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch_geometric.utils as tu
from torch_geometric.data import Data
from .gpn_fns.nn import uce_loss, entropy_reg
from .gpn_fns.layers import APPNPPropagation, LinearSequentialLayer
from .gpn_fns.utils import apply_mask
from .gpn_fns.utils import Prediction, ModelConfiguration
from .gpn_fns.layers import Density, Evidence, ConnectedComponents
from .gpn_fns.model import Model
import numpy as np

from ...utils.utils import config_to_dict, get_statistics, get_device, get_edge_indices, escape_slashes
import os
import copy


class GPN(Model):
    """Graph Posterior Network model"""

    def __init__(self, params: ModelConfiguration, graph, training_labels, config):
        super().__init__(params)

        if self.params.num_layers is None:
            num_layers = 0

        else:
            num_layers = self.params.num_layers

        if num_layers > 2:
            self.input_encoder = LinearSequentialLayer(
                self.params.dim_features,
                [self.params.dim_hidden] * (num_layers - 2),
                self.params.dim_hidden,
                batch_norm=self.params.batch_norm,
                dropout_prob=self.params.dropout_prob,
                activation_in_all_layers=True)
        else:
            self.input_encoder = nn.Sequential(
                nn.Linear(self.params.dim_features, self.params.dim_hidden),
                nn.ReLU(),
                nn.Dropout(p=self.params.dropout_prob))

        self.latent_encoder = nn.Linear(self.params.dim_hidden, self.params.dim_latent)

        use_batched = True if self.params.use_batched_flow else False 
        self.flow = Density(
            dim_latent=self.params.dim_latent,
            num_mixture_elements=self.params.num_classes,
            radial_layers=self.params.radial_layers,
            maf_layers=self.params.maf_layers,
            gaussian_layers=self.params.gaussian_layers,
            use_batched_flow=use_batched)

        self.evidence = Evidence(scale=self.params.alpha_evidence_scale)

        self.propagation = APPNPPropagation(
            K=self.params.K,
            alpha=self.params.alpha_teleport,
            add_self_loops=self.params.add_self_loops,
            cached=False,
            normalization='sym')
        
        self.graph = graph
        self.edge_indices, self.edge_weights = get_edge_indices(self.graph.adj_matrix)
        self.training_labels = training_labels
        self.config = config

        assert self.params.pre_train_mode in ('encoder', 'flow', None)
        assert self.params.likelihood_type in ('UCE', 'nll_train', 'nll_train_and_val', 'nll_consistency', None)

    def forward(self, attr_matrix: torch.sparse.FloatTensor, idx: torch.LongTensor, y = None) -> Prediction:
        return self.forward_impl(attr_matrix, idx, y)

    def forward_impl(self, attr_matrix: torch.sparse.FloatTensor, idx: torch.LongTensor, y = None) -> Prediction:
        edge_index = self.edge_indices
        h = self.input_encoder(attr_matrix)
        z = self.latent_encoder(h)

        # compute feature evidence (with Normalizing Flows)
        # log p(z, c) = log p(z | c) p(c)
        p_c = self.get_class_probalities(attr_matrix, idx)
        log_q_ft_per_class = self.flow(z) + p_c.view(1, -1).log()

        if '-plus-classes' in self.params.alpha_evidence_scale:
            further_scale = self.params.num_classes
        else:
            further_scale = 1.0

        beta_ft = self.evidence(
            log_q_ft_per_class, dim=self.params.dim_latent,
            further_scale=further_scale).exp()

        alpha_features = 1.0 + beta_ft

        beta = self.propagation(beta_ft, edge_index)
        alpha = 1.0 + beta

        soft_iso = alpha_features / alpha_features.sum(-1, keepdim=True)
        soft = alpha / alpha.sum(-1, keepdim=True)
        logits = None
        
        log_soft = soft.log()
        
        max_soft_iso, _ = soft_iso.max(dim=-1)
        max_soft, hard = soft.max(dim=-1)

        # ---------------------------------------------------------------------------------
        pred = Prediction(
            # predictions and intermediary scores
            alpha=alpha,
            soft=soft,
            log_soft=log_soft,
            hard=hard,

            logits=logits,
            latent=z,
            latent_features=z,

            hidden=h,
            hidden_features=h,

            evidence=beta.sum(-1),
            evidence_ft=beta_ft.sum(-1),
            log_ft_per_class=log_q_ft_per_class,

            # prediction confidence scores
            prediction_confidence_aleatoric=max_soft,
            prediction_confidence_epistemic=alpha[torch.arange(hard.size(0)), hard],
            prediction_confidence_structure=None,

            # sample confidence scores
            sample_confidence_aleatoric=max_soft,
            sample_confidence_aleatoric_isolated=max_soft_iso,
            sample_confidence_epistemic=alpha.sum(-1),
            sample_confidence_features=alpha_features.sum(-1),
            sample_confidence_structure=None
        )
        # ---------------------------------------------------------------------------------

        return pred

    def get_optimizer(self, lr: float, weight_decay: float) -> List[optim.Adam]:
        flow_lr = lr if self.params.factor_flow_lr is None else self.params.factor_flow_lr * lr
        flow_weight_decay = weight_decay if self.params.flow_weight_decay is None else self.params.flow_weight_decay

        flow_params = list(self.flow.named_parameters())
        flow_param_names = [f'flow.{p[0]}' for p in flow_params]
        flow_param_weights = [p[1] for p in flow_params]

        all_params = list(self.named_parameters())
        params = [p[1] for p in all_params if p[0] not in flow_param_names]

        # all params except for flow
        flow_optimizer = optim.Adam(flow_param_weights, lr=flow_lr, weight_decay=flow_weight_decay)
        model_optimizer = optim.Adam(
            [{'params': flow_param_weights, 'lr': flow_lr, 'weight_decay': flow_weight_decay},
             {'params': params}],
            lr=lr, weight_decay=weight_decay)

        return [model_optimizer, flow_optimizer]

    def get_warmup_optimizer(self, lr: float, weight_decay: float) -> List[optim.Adam]:
        model_optimizer, flow_optimizer = self.get_optimizer(lr, weight_decay)

        if self.params.pre_train_mode == 'encoder':
            warmup_optimizer = model_optimizer
        else:
            warmup_optimizer = flow_optimizer

        return [warmup_optimizer]

    def get_finetune_optimizer(self, lr: float, weight_decay: float) -> List[optim.Adam]:
        # similar to warmup
        return [self.get_warmup_optimizer(lr, weight_decay)]
    
    def set_optimizers(self):
        self.optimizers = []
        for era in range(self.config.number_of_eras):
            if self.config.optimizers[era] == "warmup":
                optimizer_list = self.get_warmup_optimizer(self.config.learning_rate[era], self.config.reg_lambda[era])
            elif self.config.optimizers[era] == "training":
                optimizer_list = self.get_optimizer(self.config.learning_rate[era], self.config.reg_lambda[era])
            elif self.config.optimizers[era] == "finetune":
                optimizer_list = self.get_finetune_optimizer(self.config.learning_rate[era], self.config.reg_lambda[era])
            else:
                raise NotImplementedError()
            self.optimizers.append(optimizer_list)

    def uce_loss(self, prediction: Prediction, idx, labels, approximate=True) -> Tuple[torch.Tensor, torch.Tensor]:
        #alpha_train, y = apply_mask(idx, labels, prediction.alpha, split='train')
        alpha_train = prediction.alpha[idx]
        reg = self.params.entropy_reg
        return uce_loss(alpha_train, labels, reduction='sum'), \
            entropy_reg(alpha_train, reg, approximate=approximate, reduction='sum')

    def loss(self, prediction: Prediction, idx, labels) -> Dict[str, torch.Tensor]:
        uce, reg = self.uce_loss(prediction, idx, labels)
        n_train = idx.shape[0] if self.params.loss_reduction == 'mean' else 1
        return {'UCE': uce / n_train, 'REG': reg / n_train}

    def warmup_loss(self, prediction: Prediction, idx, labels) -> Dict[str, torch.Tensor]:
        if self.params.pre_train_mode == 'encoder':
            return self.CE_loss(prediction, idx, labels)

        return self.loss(prediction, idx, labels)

    def finetune_loss(self, prediction: Prediction, idx, labels) -> Dict[str, torch.Tensor]:
        return self.warmup_loss(prediction, idx, labels)

    def likelihood(self, prediction: Prediction, idx, labels) -> Dict[str, torch.Tensor]:
        raise NotImplementedError

    def get_class_probalities(self, attr_matrix: torch.sparse.FloatTensor, idx: torch.LongTensor) -> torch.Tensor:
        l_c = torch.zeros(self.params.num_classes, device=attr_matrix.device)
        y_train = self.training_labels

        # calculate class_counts L(c)
        for c in range(self.params.num_classes):
            class_count = (y_train == c).sum()
            l_c[c] = class_count

        L = l_c.sum()
        p_c = l_c / L

        return p_c
    
    # Compability Functions
    def get_predictions(self, attr_matrix, idx):
        prediction = self.forward(attr_matrix, idx)

        epist = prediction.sample_confidence_epistemic[idx] if self.config.network_effects else prediction.sample_confidence_features[idx]

        alea = prediction.sample_confidence_aleatoric[idx] if self.config.network_effects else prediction.sample_confidence_aleatoric_isolated[idx]

        return prediction.soft[idx], prediction.hard[idx], epist, alea
    
    def training_init(self, pytorch_seed, date_time_str, iteration, data_seed):
        torch.manual_seed(seed=pytorch_seed)
        self.set_optimizers()
        self.pytorch_seed = pytorch_seed
        self.date_time_str = date_time_str
        self.iteration = iteration
        self.data_seed = data_seed
    
    def get_loss(self, attr_mat_norm, idx, labels, oods):
        self.set_eval(True)
        id_idx = idx[oods==0]
        id_labels = labels[oods==0]
        prediction = self.forward(attr_mat_norm, id_idx) 
        
        loss = torch.zeros([1], device=attr_mat_norm.device)
        for key, val in self.loss(prediction, id_idx, id_labels).items():
            loss += val
        
        return loss
    
    def training_step(self, phase, attr_mat_norm, idx, labels, oods):
        self.set_eval(phase != "train")
        self.optimizers_zero_grad()
        
        id_idx = idx[oods==0]
        id_labels = labels[oods==0]
        prediction = self.forward(attr_mat_norm, id_idx) 
        loss = torch.zeros([1], device=attr_mat_norm.device)
        
        if phase == "train":
            loss_dict = self.loss(prediction, id_idx, id_labels)
            for key, val in loss_dict.items():
                loss += val
            loss.backward()
            self.optimizers_step()   
        else:
            loss = F.nll_loss(prediction.log_soft[id_idx], id_labels, reduction=self.params.loss_reduction).cpu().detach()    
        
        probs, preds, confidence_all, confidence_correct, confidence_false, datapoints_correct, datapoints_false = get_statistics(self, attr_mat_norm, idx, labels, oods)
        
        return loss, probs, confidence_all, confidence_correct, confidence_false, datapoints_correct, datapoints_false
    
    def optimizers_zero_grad(self):
        optimizers_for_era = self.optimizers[self.era]
        for optimizer in optimizers_for_era:
            optimizer.zero_grad()

    def optimizers_step(self):
        optimizers_for_era = self.optimizers[self.era]
        for optimizer in optimizers_for_era:
            optimizer.step()
    
    def custom_state_dict(self, acc):
        return {
            'model': copy.deepcopy(self.state_dict()),
            'config': copy.deepcopy(config_to_dict(self.config)),
            'stopping_acc': copy.deepcopy(acc),
            'torch_seed': copy.deepcopy(self.pytorch_seed),
            'data_seed': copy.deepcopy(self.data_seed),
            'iteration': copy.deepcopy(self.iteration),
            'date_time': copy.deepcopy(self.date_time_str)
        }

    def save_model(self, custom_state_dict = None):
        if custom_state_dict is None:
            custom_state_dict = self.custom_state_dict(None)
        torch.save(custom_state_dict, escape_slashes(os.getcwd() + '/saved_models/' +  'GPN [' + self.date_time_str + '] [' + str(self.data_seed) + '] [' + str(self.iteration) + ']' + '.save'))
    
    def load_model_from_state_dict(self, state_dict, attr_mat):
        self.training_init(state_dict["torch_seed"], state_dict["date_time"], state_dict["iteration"], state_dict["data_seed"])
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
    
    def log_weights(self):
        return {}
    
    def set_era(self, era):
        self.era = era
        
        if self.config.optimizers[self.era] == "warmup":
            self.set_warming_up(True)
            self.set_finetuning(False)
        elif self.config.optimizers[self.era] == "training":
            self.set_warming_up(False)
            self.set_finetuning(False)
        elif self.config.optimizers[self.era] == "finetune":
            self.set_warming_up(False)
            self.set_finetuning(True)
        else:
            raise NotImplementedError()
    
        