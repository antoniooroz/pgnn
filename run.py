##############################################################
# This file is based on the reproduce_results_pytorch.ipynb file from the following source
# Author: Johannes Gasteiger, Aleksandar Bojchevski and Stephan Günnemann
# Last Visited: 14.06.2022
# Title: PPNP and APPNP
# URL: https://github.com/gasteigerjo/ppnp
##############################################################

import logging

import wandb
from datetime import datetime

import pgnn.models as models
from pgnn.training import train_model
import pgnn.utils.earlystopping as ES
from pgnn.data.io import load_dataset

from pgnn.logger import Logger

import pgnn.utils.stat_helpers as stat_helpers

import copy
import  pgnn.utils.arguments_parsing as arguments_parsing

import numpy as np

from pgnn.utils import get_device


def run(seml_args=None, sweep=False):
    logging.basicConfig(
        format='%(asctime)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        level=logging.INFO + 2)
     
    logging.info(f'SEML args: {seml_args}')
    
    args = arguments_parsing.parse_args()
    arguments_parsing.overwrite_with_seml_args(args, seml_args)
    arguments_parsing.overwrite_with_config_args(args)

    graph_name = args.dataset
    graph = load_dataset(graph_name)
    graph.standardize(select_lcc=True)

    test = True

    test_seeds = [
            2144199730,  794209841, 2985733717, 2282690970, 1901557222,
            2009332812, 2266730407,  635625077, 3538425002,  960893189,
            497096336, 3940842554, 3594628340,  948012117, 3305901371,
            3644534211, 2297033685, 4092258879, 2590091101, 1694925034]

    val_seeds = [
            2413340114, 3258769933, 1789234713, 2222151463, 2813247115,
            1920426428, 4272044734, 2092442742, 841404887, 2188879532,
            646784207, 1633698412, 2256863076,  374355442,  289680769,
            4281139389, 4263036964,  900418539,  119332950, 1628837138]

    if test:
        seeds = test_seeds
    else:
        seeds = val_seeds

    nknown = graph.adj_matrix.shape[0]
        
    idx_split_args = {'ntrain_per_class': 20, 'nstopping': 500, 'nknown': nknown}

    if graph_name == 'microsoft_academic':
        args.alpha = 0.2
    else:
        args.alpha = 0.1

    save_result = False
    print_interval = 100
    device = get_device()

    # Run
    model_name = datetime.now().strftime("%Y-%m-%d %H:%M:%S") if args.custom_name is None else args.custom_name

    #W&B 
    logger = Logger(model_name, args.mode)
    
    # Hyperparams    
    config = wandb.config
    for key in args.__dict__.keys():
        config[key] = args.__dict__[key]
        
    model_args = {
        'hiddenunits': config.hidden_units,
        'drop_prob': config.drop_prob}
        
    reg_lambda = config.reg_lambda
    
    if config.seeds:
        seeds = config.seeds
    else:
        seeds = seeds[:config.seeds_num]
        seeds = seeds[config.seeds_start:]
    
    stopping_args_single = ES.stopping_args
    stopping_args = []
    if not config.stopping_var=="acc":
        if config.stopping_var=="loss":
            stopping_args_single["stop_varnames"] = [ES.StopVariable.LOSS]
        elif config.stopping_var=="ood":
            stopping_args_single["stop_varnames"] = [ES.StopVariable.OOD, ES.StopVariable.ACCURACY, ES.StopVariable.LOSS]
        elif config.stopping_var=="ood_wo_network":
            stopping_args_single["stop_varnames"] = [ES.StopVariable.OOD_WO_NETWORK, ES.StopVariable.ACCURACY, ES.StopVariable.LOSS]
        else:
            raise NotImplementedError()
    
    for era in range(config.number_of_eras):
        stopping_args_for_era = copy.deepcopy(stopping_args_single)
        stopping_args_for_era["max_epochs"] = config.max_epochs[era]
        stopping_args_for_era["patience"] = config.patience[era]
        stopping_args.append(stopping_args_for_era)

    # W&B Logs
    max_steps = 0

    train_losses = [] #2D array [tain_losses_seed1_iter1, ...]
    stopping_losses = [] #2D array [stopping_losses_seed1_iter1, ...]

    train_durations = [] #2D array
    stopping_durations = [] #2D array

    train_accs = [] #2D array
    train_confs_correct = [] #2D array
    train_confs_false = [] #2D array
    train_confs_all = [] #2D array

    stopping_accs = [] #2D array
    stopping_confs_correct = [] #2D array
    stopping_confs_false = [] #2D array
    stopping_confs_all = [] #2D array
    
    # OOD
    ood_train = [] # 2D array
    ood_stopping = [] #2D array
    ood_final_train = [] #1D array
    ood_final_stopping = [] #1D array
    ood_final_valtest = [] #1D array

    if config.mode=="PPNP":
        model_class = models.PPNP
    elif config.mode=="MCD-PPNP":
        model_class = models.MCD_PPNP
    elif config.mode in ["P-PPNP", "Mixed-PPNP"]:
        model_class = models.P_PPNP
    elif config.mode in ["GCN", "DropEdge"]:
        model_class = models.GCN
    elif config.mode in ["MCD-GCN", "DE-GCN"]:
        model_class = models.MCD_GCN
    elif config.mode in ["P-GCN", "Mixed-GCN"]:
        model_class = models.P_GCN
    elif config.mode=="GPN":
        model_class = models.GPN
    elif config.mode=="GAT":
        model_class = models.GAT
    elif config.mode=="MCD-GAT":
        model_class = models.MCD_GAT
    elif config.mode in ["P-GAT", "P-PROJ-GAT", "P-ATT-GAT", "Mixed-GAT", "Mixed-PROJ-GAT", "Mixed-ATT-GAT"]:
        model_class = models.P_GAT
    else:
        raise NotImplementedError()

    results = []
    niter_tot = config.iters_per_seed * len(seeds)
    i_tot = 0
    for seed in seeds:
        idx_split_args['seed'] = seed
        for iteration in range(config.iters_per_seed):            
            torch_seed = None
            
            i_tot += 1
            logging_string = f"Iteration {i_tot} of {niter_tot}"
            logging.log(22,
                    logging_string + "\n                     "
                    + '-' * len(logging_string))
            logger.newIteration(seed, iteration)
            train_model(
                graph_name, model_class, graph, model_args, reg_lambda, model_name, iteration, config, logger,
                idx_split_args, stopping_args, test, device, torch_seed, print_interval)
            logger.finishIteration()
            # W&B table
            """
            # For multi-line plots
            if max_steps < len(wandb_logs["train_loss"]):
                max_steps = len(wandb_logs["train_loss"])
            
            run_names.append(run_name)
            train_losses.append(wandb_logs["train_loss"])
            stopping_losses.append(wandb_logs["stopping_loss"])
            train_durations.append(wandb_logs["train_duration"])
            stopping_durations.append(wandb_logs["stopping_duration"])
            
            train_accs.append(wandb_logs["train_acc"])
            train_confs_correct.append(wandb_logs["train_conf_correct"])
            train_confs_false.append(wandb_logs["train_conf_false"])
            train_confs_all.append(wandb_logs["train_conf_all"])
            
            stopping_accs.append(wandb_logs["stopping_acc"])
            stopping_confs_correct.append(wandb_logs["stopping_conf_correct"])
            stopping_confs_false.append(wandb_logs["stopping_conf_false"])
            stopping_confs_all.append(wandb_logs["stopping_conf_all"])
            
            # OOD
            if config.ood != "none":
                ood_train.append(wandb_logs["ood_train"])
                ood_stopping.append(wandb_logs["ood_stopping"])
                ood_final_train.append(wandb_logs["ood_final_train"])
                ood_final_stopping.append(wandb_logs["ood_final_stopping"])
                ood_final_valtest.append(wandb_logs["ood_final_valtest"])
                
            """
    """
    # Create and log statistics
    final_stopping_stats = stat_helpers.get_stats_for_column(results_table, "final_stopping_acc", "final/stopping_acc")
    final_train_stats = stat_helpers.get_stats_for_column(results_table, "final_train_acc", "final/train_acc")
    final_valtest_stats = stat_helpers.get_stats_for_column(results_table, "final_valtest_acc", "final/valtest_acc")
    # Loss
    final_valtest_loss = stat_helpers.get_stats_for_column(results_table, "final_valtest_loss", "final_loss/valtest_loss")
    final_train_loss = stat_helpers.get_stats_for_column(results_table, "final_train_loss", "final_loss/train_loss")
    final_stopping_loss = stat_helpers.get_stats_for_column(results_table, "final_stopping_loss", "final_loss/stopping_loss")

    ood_logs = {}
    if config.ood != "none":
        ood_logs_graphs = {
            # Averaged Training Graphs
            **stat_helpers.ood_average_stats("train", ood_train, max_steps),
            **stat_helpers.ood_average_stats("stopping", ood_stopping, max_steps)
        }
        ood_logs_stats = {
            # Final Results
            **stat_helpers.ood_final_stats("train", ood_final_train),
            **stat_helpers.ood_final_stats("stopping", ood_final_stopping),
            **stat_helpers.ood_final_stats("valtest", ood_final_valtest)
        }
        ood_logs = {
            **ood_logs_graphs,
            **ood_logs_stats
        }

    training_stats = {
        "losses": train_losses,
        "accs": train_accs,
        "confs_all": train_confs_all,
        "confs_correct": train_confs_correct,
        "confs_false": train_confs_false,
        "max_steps": max_steps
    }

    stopping_stats = {
        "losses": stopping_losses,
        "accs": stopping_accs,
        "confs_all": stopping_confs_all,
        "confs_correct": stopping_confs_correct,
        "confs_false": stopping_confs_false,
        "max_steps": max_steps
    }
    
    if config.skip_training:
        training_stats = {}
    else:
        training_stats = {
            # Multiline Graphs
            "multiline/train_losses": wandb.plot.line_series(xs = range(max_steps), ys = train_losses, keys = run_names, title = "Train Losses (Multiline)", xname="step"),
            "multiline/stopping_losses": wandb.plot.line_series(xs = range(max_steps), ys = stopping_losses, keys = run_names, title = "Stopping Losses (Multiline)", xname="step"),
            "multiline/train_durations": wandb.plot.line_series(xs = range(max_steps), ys = train_durations, keys = run_names, title = "Train Durations (Multiline)", xname="step"),
            "multiline/stopping_durations": wandb.plot.line_series(xs = range(max_steps), ys = stopping_durations, keys = run_names, title = "Train Durations (Multiline)", xname="step"),
            "multiline/stopping_accs": wandb.plot.line_series(xs = range(max_steps), ys = stopping_accs, keys = run_names, title = "Stopping Accuracies (Multiline)", xname="step"),
            "multiline/stopping_confs_correct": wandb.plot.line_series(xs = range(max_steps), ys = stopping_confs_correct, keys = run_names, title = "Stopping Confidence Correct (Multiline)", xname="step"),
            "multiline/stopping_confs_false": wandb.plot.line_series(xs = range(max_steps), ys = stopping_confs_false, keys = run_names, title = "Stopping Confidence False (Multiline)", xname="step"),
            "multiline/stopping_confs_all": wandb.plot.line_series(xs = range(max_steps), ys = stopping_confs_all, keys = run_names, title = "Stopping Confidence All (Multiline)", xname="step"),
            # Accumulated Graphs Train
            **stat_helpers.accumulated_stats(stats=training_stats, prefix="train", mode="mean"),
            **stat_helpers.accumulated_stats(stats=training_stats, prefix="train", mode="max"),
            **stat_helpers.accumulated_stats(stats=training_stats, prefix="train", mode="min"),
            # Accumulated Graphs Stopping
            **stat_helpers.accumulated_stats(stats=stopping_stats, prefix="stopping", mode="mean"),
            **stat_helpers.accumulated_stats(stats=stopping_stats, prefix="stopping", mode="max"),
            **stat_helpers.accumulated_stats(stats=stopping_stats, prefix="stopping", mode="min")
        }

    other_stats = {
        **ood_logs,
        "results": results_table,
        # Accuracies per model
        "final_per_model_plot/stopping_acc": wandb.plot.bar(results_table, "run_name", "final_stopping_acc", title="Final Stopping Accuracy Per Model"),
        "final_per_model_plot/train_acc": wandb.plot.bar(results_table, "run_name", "final_train_acc", title="Final Training Accuracy Per Model"),
        "final_per_model_plot/valtest_acc": wandb.plot.bar(results_table, "run_name", "final_valtest_acc", title="Final Valtest Accuracy Per Model"),
        # Confidences per model
        "final_per_model_plot/stopping_conf_correct": wandb.plot.bar(results_table, "run_name", "final_stopping_conf_correct", title="Final Stopping Confidence Correct"),
        "final_per_model_plot/stopping_conf_false": wandb.plot.bar(results_table, "run_name", "final_stopping_conf_false", title="Final Stopping Confidence False"),
        "final_per_model_plot/stopping_conf_all": wandb.plot.bar(results_table, "run_name", "final_stopping_conf_all", title="Final Stopping Confidence All"),
        "final_per_model_plot/train_conf_correct": wandb.plot.bar(results_table, "run_name", "final_train_conf_correct", title="Final Training Confidence Correct"),
        "final_per_model_plot/train_conf_false": wandb.plot.bar(results_table, "run_name", "final_train_conf_false", title="Final Training Confidence False"),
        "final_per_model_plot/train_conf_all": wandb.plot.bar(results_table, "run_name", "final_train_conf_all", title="Final Training Confidence All"),
        "final_per_model_plot/valtest_conf_correct": wandb.plot.bar(results_table, "run_name", "final_valtest_conf_correct", title="Final Valtest Confidence Correct"),
        "final_per_model_plot/valtest_conf_false": wandb.plot.bar(results_table, "run_name", "final_valtest_conf_false", title="Final Valtest Confidence False"),
        "final_per_model_plot/valtest_conf_all": wandb.plot.bar(results_table, "run_name", "final_valtest_conf_all", title="Final Valtest Confidence All"),
        # Binned accuracies
        "final_binned_plot/stopping_acc": wandb.plot.histogram(results_table, "final_stopping_acc", title="Final Stopping Accuracies (Binned)"),
        "final_binned_plot/final_train_acc": wandb.plot.histogram(results_table, "final_train_acc", title="Final Training Accuracies (Binned)"),
        "final_binned_plot/final_valtest_acc": wandb.plot.histogram(results_table, "final_valtest_acc", title="Final Valtest Accuracies (Binned)"),
        **training_stats
    }

    log_dictionary = {**final_stopping_stats, **final_train_stats, **final_valtest_stats, **other_stats, **final_valtest_loss, **final_stopping_loss, **final_train_loss}
    wandb.log(log_dictionary)
    """

    logger.finish()
    
        
if __name__ == "__main__":
    run()
    

    