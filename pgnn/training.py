##############################################################
# This file is a modified version from the following source
# Author: Johannes Gasteiger, Aleksandar Bojchevski and Stephan Günnemann
# Last Visited: 14.06.2022
# Title: PPNP and APPNP
# URL: https://github.com/gasteigerjo/ppnp
##############################################################

from typing import Type, Tuple
import time
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from pgnn.base.propagation import PPRPowerIterationAlternative, AttentionPropagation, AttentionPropagation2
from pgnn.logger import Logger
import tqdm

import copy
import os
from pgnn.models.gpn.gpn_fns.utils.config import ModelConfiguration

from pgnn.ood import OOD_Experiment
from pgnn.base import PPRPowerIteration

from .data.sparsegraph import SparseGraph
from .preprocessing import gen_seeds, gen_splits, normalize_attributes
from pgnn.utils import EarlyStopping, stopping_args, get_device, matrix_to_torch, final_run

import pyro
from pyro.infer.autoguide import AutoDiagonalNormal
from pyro.infer import SVI, Trace_ELBO, Predictive

def get_dataloaders(idx, labels_np, oods_all, batch_size=None):
    labels = torch.LongTensor(labels_np)
    if batch_size is None:
        batch_size = max((val.numel() for val in idx.values()))
    datasets = {phase: TensorDataset(ind, labels[ind], oods_all[ind]) for phase, ind in idx.items()}
    dataloaders = {phase: DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)
                   for phase, dataset in datasets.items()}
    return dataloaders


def train_model(
        name: str, model_class: Type[nn.Module], graph: SparseGraph, model_args: dict,
        reg_lambda: float, model_name: str, iteration: int, config, logger: Logger,
        idx_split_args: dict = {'ntrain_per_class': 20, 'nstopping': 500, 'nknown': 1500, 'seed': 2413340114},
        stopping_args: list = [stopping_args],
        test: bool = False, device: str = get_device(),
        torch_seed: int = None, print_interval: int = 10) -> Tuple[nn.Module, dict]:
    labels_all = graph.labels.astype('int64')
    idx_np = {}
    idx_np['train'], idx_np['stopping'], idx_np['valtest'] = gen_splits(labels_all, idx_split_args, test=False)
    
    idx_all = {key: torch.LongTensor(val) for key, val in idx_np.items()}
    oods_all = torch.zeros(labels_all.shape)
    
    attr_matrix = graph.attr_matrix.copy()
    
    if config.binary_attributes:
        attr_matrix[attr_matrix > 0] = 1

    if config.normalize_attributes == 'default':
        attr_mat_norm_np = normalize_attributes(attr_matrix)
        attr_mat_norm = matrix_to_torch(attr_mat_norm_np).to(device)
    elif config.normalize_attributes == 'div_by_sum':
        attr_mat_norm = matrix_to_torch(attr_matrix).to(device)
        attr_mat_norm = attr_mat_norm / (attr_mat_norm.sum(dim=-1).unsqueeze(-1) + 1e-10)
    elif config.normalize_attributes == 'no':
        attr_mat_norm = matrix_to_torch(attr_matrix).to(device)
    else:
        raise NotImplementedError()
    
    # OOD
    ood_experiment = None
    if config.ood != "none":
        ood_experiment = OOD_Experiment(config, graph.adj_matrix, attr_mat_norm, idx_all, labels_all, oods_all)
        adj_matrix, attr_mat_norm, idx_all, labels_all, oods_all = ood_experiment.setup()
    else:
        adj_matrix = graph.adj_matrix
        

    logging.log(21, f"{model_class.__name__}: {model_args}")
    if torch_seed is None:
        torch_seed = gen_seeds()
    torch.manual_seed(seed=torch_seed)
    logging.log(22, f"PyTorch seed: {torch_seed}")

    nfeatures = attr_matrix.shape[1]
    
    if config.remove_loc_classes:
        nclasses = max(labels_all[oods_all==0]) + 1
    else:
        nclasses = max(labels_all) + 1
        
    if config.mode=="PPNP" or config.mode=="MCD-PPNP" or config.mode=="P-PPNP" or config.mode=="Mixed-PPNP":
        #prop_appnp = AttentionPropagation(adj_matrix, alpha=config.alpha, niter=10, feature_size=nfeatures)
        prop_appnp = PPRPowerIteration(adj_matrix, alpha=config.alpha, niter=10)
        model_args = {**model_args, "nfeatures": nfeatures, "nclasses": nclasses, "config": config, "propagation": prop_appnp}
    elif config.mode in ["GCN", "MCD-GCN", "P-GCN", "Mixed-GCN", "DE-GCN"]:
        model_args = {**model_args, "nfeatures": nfeatures, "nclasses": nclasses, "config": config, "adj_matrix": adj_matrix}
    elif config.mode=="GPN":
        config.gpn_model["dim_features"]=nfeatures
        config.gpn_model["num_classes"]=nclasses
        model_args = {"params": ModelConfiguration(**config.gpn_model), "graph": graph, "training_labels": labels_all[idx_all["train"]], "config": config}
    elif config.mode in ["GAT", "P-GAT", "P-PROJ-GAT", "P-ATT-GAT", "Mixed-GAT", "Mixed-PROJ-GAT", "Mixed-ATT-GAT", "MCD-GAT"]:
        model_args = {"nfeatures": nfeatures, "nclasses": nclasses, "config": config, "adj_matrix": adj_matrix}
    else:
        raise NotImplementedError()
    
    model = model_class(**model_args).to(device)

    logger.watch(model)
    
    model.training_init(torch_seed, model_name, iteration, idx_split_args['seed'])

    dataloaders = get_dataloaders(idx_all, labels_all, oods_all)
    
    early_stopping_list = []
    for era in range(config.number_of_eras):
        early_stopping_list.append(EarlyStopping(model, **stopping_args[era]))
    
    if ood_experiment is not None:
        ood_experiment.setup_dataloaders(dataloaders)

    epoch_stats = {'train': {}, 'stopping': {}}

    start_time = time.time()
    last_time = start_time
    
    best_stopping_value = None
    best_model_parameters = None

    pyro.clear_param_store()
    
    model.set_era(0)
    early_stopping = early_stopping_list[0]

    if config.load is not None:
        model.load_model(config.mode + ' [' + config.load + '] [' + str(idx_split_args['seed']) + '] [' + str(iteration) + ']' + '.save', attr_mat_norm)
        
    if not config.skip_training:
        for era in range(config.number_of_eras):
            early_stopping = early_stopping_list[era]
            pbar = tqdm.tqdm(range(early_stopping.max_epochs))
            model.set_era(era)
            for epoch in pbar:
                for phase in epoch_stats.keys():
                    running_loss = torch.tensor(0.0).to(get_device())
                    running_confidence_all = torch.tensor(0.0).to(get_device())
                    running_confidence_correct = torch.tensor(0.0).to(get_device())
                    running_confidence_false = torch.tensor(0.0).to(get_device())
                    running_datapoints_correct = torch.tensor(0.0).to(get_device())
                    running_datapoints_false = torch.tensor(0.0).to(get_device())
                    running_probs = []
                    
                    ########################################################################
                    # Training Step                                                        #
                    ########################################################################
                    for idx, labels, oods in dataloaders[phase]:
                        idx = idx.to(device)
                        labels = labels.to(device)

                        loss, probs, confidence_all, confidence_correct, confidence_false, datapoints_correct, datapoints_false = model.training_step(phase if epoch > 0 else "before_training", attr_mat_norm, idx, labels, oods)
                        
                        running_loss += loss * idx.size(0)

                        running_confidence_all += confidence_all
                        running_confidence_correct += confidence_correct
                        running_confidence_false += confidence_false
                        running_datapoints_correct += datapoints_correct
                        running_datapoints_false += datapoints_false
                        running_probs.append(probs)
                        
                    ########################################################################
                    # Logging                                                              #
                    ########################################################################
                    duration = time.time() - last_time
                    
                    log_loss =  (running_loss / (running_datapoints_correct+running_datapoints_false)).item()
                    log_acc = (running_datapoints_correct / (running_datapoints_correct+running_datapoints_false)).item()
                    log_conf_correct = (running_confidence_correct / running_datapoints_correct).item()
                    log_conf_false = (running_confidence_false / running_datapoints_false).item()
                    log_conf_all = (running_confidence_all / (running_datapoints_correct+running_datapoints_false)).item()
                    log_duration = duration
                    log_seed = idx_split_args['seed']
                    log_iteration = iteration        

                    ########################################################################
                    # OOD Logs                                                             #
                    ########################################################################
                    ood_logs = None
                    if ood_experiment and config.ood_eval_during_training:
                        ood_logs = ood_experiment.run(model, device, phase)
                        
                        epoch_stats[phase]['ood'] = ood_logs["auc_roc_score"]
                        epoch_stats[phase]['ood_wo_network'] = ood_logs["auc_roc_score_wo_network_effects"]
                    
                    ########################################################################
                    # Saving Logs                                                          #
                    ########################################################################

                    epoch_stats[phase]['loss'] = log_loss
                    epoch_stats[phase]['acc'] = log_acc
                    
                    logger.logStep(
                        phase=phase, 
                        logs={
                            "loss": log_loss,
                            "acc": log_acc,
                            "conf_correct": log_conf_correct,
                            "conf_false": log_conf_false,
                            "conf_all": log_conf_all,
                            "duration": log_duration,
                            "seed": log_seed,
                            "iter": log_iteration,
                        }, 
                        ood=ood_logs,
                        weights=model.log_weights()
                    )
                        
                    last_time = time.time()
                
                pbar.set_postfix({'stopping_acc': epoch_stats['stopping']['acc']})
                ########################################################################
                # Early Stopping                                                       #
                ########################################################################
                if config.use_early_stopping[era]:
                    if epoch_stats['stopping'][early_stopping.stop_vars[0]] != None and (best_stopping_value == None or early_stopping.comp_ops[0](epoch_stats['stopping'][early_stopping.stop_vars[0]], best_stopping_value)):
                        best_stopping_value = epoch_stats['stopping'][early_stopping.stop_vars[0]]
                        best_model_parameters = model.custom_state_dict(best_stopping_value)
                
                    if len(early_stopping.stop_vars) > 0:
                        stop_vars = [epoch_stats['stopping'][key]
                                    for key in early_stopping.stop_vars]
                        if early_stopping.check(stop_vars, epoch):
                            break
                    
                    
            # Load best model parameters from era
            if config.use_early_stopping[era]:
                model.load_model_from_state_dict(best_model_parameters, attr_mat_norm)
            else:
                best_stopping_value = epoch_stats['stopping'][early_stopping.stop_vars[0]]
                best_model_parameters = model.custom_state_dict(best_stopping_value)

        runtime = time.time() - start_time
        runtime_perepoch = runtime / (epoch + 1)
        
        # Save best model
        model.save_model(best_model_parameters)
        model.load_model_from_state_dict(best_model_parameters, attr_mat_norm)
        
        pbar.close()
    else:
        epoch, early_stopping.best_epoch, runtime, runtime_perepoch = 0, 0, 0, 0

    ########################################################################
    # After Training                                                       #
    ########################################################################
    model.set_eval()
    
    # Stats
    train_stats = final_run(model, attr_mat_norm, idx_all['train'], labels_all, oods_all)
    stopping_stats = final_run(model, attr_mat_norm, idx_all['stopping'], labels_all, oods_all)
    valtest_stats = final_run(model, attr_mat_norm, idx_all['valtest'], labels_all, oods_all)
    
    # OOD Experiments
    ood_results_train, ood_results_stopping, ood_results_valtest = None, None, None
    if ood_experiment is not None:
        ood_results_train = ood_experiment.run(model, device, "train")
        ood_results_stopping = ood_experiment.run(model, device, "stopping")
        ood_results_valtest = ood_experiment.run(model, device, "valtest")
    
    # Logging
    weights = model.log_weights()
    logger.logEval(phase='train', logs=train_stats, ood=ood_results_train, weights=weights)
    logger.logEval(phase='stopping', logs=stopping_stats, ood=ood_results_stopping, weights=weights)
    logger.logEval(phase='valtest', logs=valtest_stats, ood=ood_results_valtest, weights=weights)
    
    logger.logAdditionalStats({
        'last_epoch': epoch,
        'best_epoch': early_stopping.best_epoch,
        'runtime': runtime,
        'runtime_per_epoch': runtime_perepoch
    })