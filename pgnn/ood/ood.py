##############################################################
# This file is a modified version from the following source
# Author: Maximilian Stadler, Bertrand Charpentier, Simon Geisler, Daniel Zügner and Stephan Günnemann
# Last Visited: 14.06.2022
# Title: Graph Posterior Network
# URL: https://github.com/stadlmax/Graph-Posterior-Network
##############################################################

import math
import torch
import numpy as np
from sklearn.metrics import roc_auc_score
from scipy import sparse

from pgnn.utils import matrix_to_torch

class OOD_Experiment(object):
    def __init__(self, config, adj_matrix, attr_mat_norm, idx_all, labels_all, oods_all):
        super().__init__()
        self.config = config
        self.idx_all = idx_all
        self.labels_all = labels_all
        self.oods_all = oods_all
        self.attr_mat_norm = attr_mat_norm
        self.adj_matrix = adj_matrix
        
    def setup(self):
        if self.config.ood == "loc":
            self._loc_setup()
        elif self.config.ood == "perturb":
            self._perturb_setup()
        elif self.config.ood == "mixed":
            self._loc_setup()
            self._perturb_setup()
            
        return self.adj_matrix, self.attr_mat_norm, self.idx_all, self.labels_all, self.oods_all
    
    def setup_dataloaders(self, dataloaders):
        self.dataloaders = dataloaders
    
    # LOC Setup
    def _loc_setup(self):
        if self.config.remove_loc_classes:
            assert self.config.loc_last_classes==True # Only allow remove loc classes if loc_last_classes
        
        ood_classes, id_classes = self._loc_get_loc_classes()

        if self.config.loc_remove_edges:
            self._loc_remove_id_ood_edges(ood_classes, id_classes)

        for c in ood_classes:
            self.oods_all[self.labels_all==c] = 1
            
    def _loc_get_loc_classes(self):
        if self.config.loc_classes is not None:
            return np.asarray(self.config.loc_classes)
        
        num_classes = self.labels_all.max() + 1
        classes = np.arange(num_classes)
        
        if not self.config.loc_last_classes:
            classes = np.random.shuffle(classes)
            
        if self.config.loc_num_classes is not None:
            num_loc_classes = self.config.loc_num_classes
        else:
            num_loc_classes = math.floor(num_classes * self.config.loc_frac)
            
        return classes[num_classes-num_loc_classes : num_classes], classes[:num_classes-num_loc_classes]

    def _loc_remove_id_ood_edges(self, ood_classes, id_classes):
        adj_matrix = matrix_to_torch(self.adj_matrix)
        for ood_class in ood_classes:
            for id_class in id_classes:
                # Remove OOD->ID edges
                oods_to_ids = adj_matrix[self.labels_all==ood_class]
                oods_to_ids[:,self.labels_all==id_class] = 0
                adj_matrix[self.labels_all==ood_class] = oods_to_ids
                # Remove ID->OOD edges
                ids_to_oods = adj_matrix[self.labels_all==id_class]
                ids_to_oods[:,self.labels_all==ood_class] = 0
                adj_matrix[self.labels_all==id_class] = ids_to_oods

        self.adj_matrix = sparse.csr_matrix(adj_matrix)

    
    # Perturb Setup
    def _perturb_setup(self):
        eps = 1e-10
        
        idx_perturbed = self._perturb_get_indices()
        
        noise = torch.zeros([idx_perturbed.shape[0], self.attr_mat_norm.shape[1]])

        self.oods_all[idx_perturbed] = 1
        self.attr_mat_norm_id = self.attr_mat_norm
        attr_matrix_ood = self.attr_mat_norm.cpu()

        if self.config.perturb_mode == "normal":
            noise = noise.normal_()
        elif self.config.perturb_mode == "normal_pn":
            noise = noise.normal_()
            noise[noise < 0] = -noise[noise < 0]
            if self.config.normalize_attributes != 'no':
                noise = noise / (noise.sum(dim=-1).unsqueeze(-1) + eps)
        elif self.config.perturb_mode == "bernoulli_0.5":
            noise = noise.bernoulli(0.5)
            if self.config.normalize_attributes != 'no':
                noise = noise / (noise.sum(dim=-1).unsqueeze(-1) + eps)
        elif self.config.perturb_mode == "bernoulli_p_calculated":
            p_calc = attr_matrix_ood.clone().detach()
            p_calc[p_calc > 0] = 1
            p = p_calc.mean().item()
            noise = noise.bernoulli(p)
            if self.config.normalize_attributes != 'no':
                noise = noise / (noise.sum(dim=-1).unsqueeze(-1) + eps)
        elif self.config.perturb_mode == "bernoulli_custom":
            noise = noise.bernoulli(self.config.perturb_custom_p)
            if self.config.normalize_attributes != 'no':
                noise = noise / (noise.sum(dim=-1).unsqueeze(-1) + eps)
        elif self.config.perturb_mode == "shuffle":
            noise = attr_matrix_ood[idx_perturbed]
            for i in range(noise.shape[0]):
                noise[i] = noise[i][torch.randperm(noise.shape[1])]
        elif self.config.perturb_mode == "zeros":
            noise *= 0
        elif self.config.perturb_mode == "ones":
            noise += 1
        else:
            raise NotImplementedError()
        
        attr_matrix_ood[idx_perturbed] = noise * self.config.perturb_noise_scale

        self.attr_mat_norm = attr_matrix_ood.to(self.attr_mat_norm_id.device)
        
    def _perturb_get_indices(self):
        def _internal_get_indices(idx):
            n_perturbed = int(idx.shape[0] * self.config.perturb_budget)
            idx_perturbed = np.random.choice(idx, n_perturbed, replace=False)
            return torch.tensor(idx_perturbed)
        
        idx_perturbed = []
        if self.config.perturb_train:
            idx_train_preturbed = _internal_get_indices(self.idx_all["train"])
            #self.idx_all["ood_train"] = idx_train_preturbed
            idx_perturbed.append(idx_train_preturbed)
            
        idx_stopping_perturbed = _internal_get_indices(self.idx_all["stopping"])
        idx_valtest_perturbed = _internal_get_indices(self.idx_all["valtest"])
        
        idx_perturbed.append(idx_stopping_perturbed)
        idx_perturbed.append(idx_valtest_perturbed)
        
        return torch.cat(idx_perturbed, dim=0)
    
    # OOD Runs
    def run(self, model, device, phase):
        model.set_eval(True)
        
        stats = self._run(model, device, phase)
        
        return stats

    def _run(self, model, device, phase):  
        # No OOD values for phase
        if self.oods_all[self.idx_all[phase]].sum() < 1:
            return {
                "ood_acc_network" : 0,
                "ood_acc_isolated" : 0,
                **self.get_ood_stats("network"),
                **self.get_ood_stats("isolated"),
                **self.get_ood_stats("mixed_network")
            }
        
        all_oods = []

        all_labels = []
        all_preds_network = []
        all_preds_isolated = []

        all_scores_epist_network = []
        all_scores_alea_network = []
        all_scores_epist_isolated = []
        all_scores_alea_isolated = []

        # OOD nodes
        for idx, labels, oods in self.dataloaders[phase]:
            epist_network, alea_network, preds_network = self._get_scores(model, device, self.attr_mat_norm, idx, True)

            epist_isolated, alea_isolated, preds_isolated = self._get_scores(model, device, self.attr_mat_norm, idx, False)

            labels = labels.cpu().detach().numpy()

            preds_network = preds_network[oods==1]
            preds_isolated = preds_isolated[oods==1]
            labels = labels[oods==1]

            all_labels.append(labels)
            all_preds_network.append(preds_network)
            all_preds_isolated.append(preds_isolated)
            
            all_scores_epist_network.append(epist_network)
            all_scores_alea_network.append(alea_network)

            all_scores_epist_isolated.append(epist_isolated)
            all_scores_alea_isolated.append(alea_isolated)

            all_oods.append(oods)

        all_scores_epist_network = np.concatenate(all_scores_epist_network, axis=0)
        all_scores_alea_network = np.concatenate(all_scores_alea_network, axis=0)

        all_scores_epist_isolated = np.concatenate(all_scores_epist_isolated, axis=0)
        all_scores_alea_isolated = np.concatenate(all_scores_alea_isolated, axis=0)

        all_scores_epist_mixed = self._get_network_mode_aggregated_score(all_scores_epist_network, all_scores_epist_isolated)
        all_scores_alea_mixed = self._get_network_mode_aggregated_score(all_scores_alea_network, all_scores_alea_isolated)

        all_preds_network = np.concatenate(all_preds_network, axis=0)
        all_preds_isolated = np.concatenate(all_preds_isolated, axis=0)

        all_labels = np.concatenate(all_labels, axis=0)

        all_oods = np.concatenate(all_oods, axis=0)        
        
        return {
            "ood_acc_network" : (all_preds_network == all_labels).mean().item(),
            "ood_acc_isolated" : (all_preds_isolated == all_labels).mean().item(),
            **self.get_ood_stats("network", all_oods, all_scores_epist_network, all_scores_alea_network),
            **self.get_ood_stats("isolated", all_oods, all_scores_epist_isolated, all_scores_alea_isolated),
            **self.get_ood_stats("mixed_network", all_oods, all_scores_epist_mixed, all_scores_alea_mixed)
        }
        
    def compute_auc_roc(self, labels, scores):
        try:
            score = roc_auc_score(labels, -scores)
            return score #if score > 0.5 else 1 - score
        except:
            return -1

    def _get_scores(self, model, device, attr_mat_norm, idx, network_effects):
        idx = idx.to(device)
        # Set network effects
        network_effects_backup = self.config.network_effects
        self.config.update({"network_effects": network_effects}, allow_val_change=True)
        # Model forward
        probs, preds, epist, alea = model.get_predictions(attr_mat_norm, idx) 
        # Reset network effects
        self.config.update({"network_effects": network_effects_backup}, allow_val_change=True)
        return epist.cpu().detach().numpy(), alea.cpu().detach().numpy(), preds.cpu().detach().numpy()
        
    def _get_network_mode_aggregated_score(self, network_scores, isolated_scores):
        network_scores = torch.from_numpy(network_scores).unsqueeze(0)
        isolated_scores = torch.from_numpy(isolated_scores).unsqueeze(0)
        
        if self.config.network_mode_uncertainty_aggregation_normalize:
            network_scores /= network_scores.abs().max() + 1e-10
            isolated_scores /= isolated_scores.abs().max() + 1e-10

        mixed_scores = torch.cat((network_scores, isolated_scores), dim=0)

        if self.config.network_mode_uncertainty_aggregation == "mean":
            return mixed_scores.mean(dim=0).numpy()
        elif self.config.network_mode_uncertainty_aggregation == "max":
            return mixed_scores.max(dim=0).values.numpy()
        elif self.config.network_mode_uncertainty_aggregation == "min":
            return mixed_scores.min(dim=0).values.numpy()
        elif self.config.network_mode_uncertainty_aggregation == "weighted_sum":
            network_scores = network_scores.squeeze(0)
            isolated_scores = isolated_scores.squeeze(0)

            network_lambda = self.config.network_mode_uncertainty_aggregation_network_lambda
            return network_lambda*network_scores + (1-network_lambda)*isolated_scores
        else:
            raise NotImplemented("Uncertainty Aggregation Mode not implemented")

    def get_ood_stats(self, postfix, oods=None, epist_scores=None, alea_scores=None):
        if oods is None or epist_scores is None or alea_scores is None:
            return {
                "auc_roc_score_"+postfix: 0,
                "auc_roc_score_alea_"+postfix : 0,
                "scores_ood_mean_"+postfix : 0,
                "scores_id_mean_"+postfix : 0,
                "scores_ood_mean_alea_"+postfix : 0,
                "scores_id_mean_alea_"+postfix : 0
            }

        auc_roc_score_epist = self.compute_auc_roc(oods, epist_scores)
        auc_roc_score_alea = self.compute_auc_roc(oods, alea_scores)

        return {
                "auc_roc_score_"+postfix: auc_roc_score_epist,
                "auc_roc_score_alea_"+postfix : auc_roc_score_alea,
                "scores_ood_mean_"+postfix : epist_scores[oods == 1].mean().item(),
                "scores_id_mean_"+postfix : epist_scores[oods == 0].mean().item(),
                "scores_ood_mean_alea_"+postfix : alea_scores[oods == 1].mean().item(),
                "scores_id_mean_alea_"+postfix : alea_scores[oods == 0].mean().item()
            }
