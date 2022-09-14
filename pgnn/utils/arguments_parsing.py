import argparse
import os
import yaml

def parse_args():
    parser = argparse.ArgumentParser(description='Train model')
    
    parser.add_argument("--dataset", type=str, default="cora_ml", help="cora_ml or citeseer")
    parser.add_argument("--config", type=str, nargs="*", default=None, help="If specified, a config file with the parameters will be used instead of the arguments given via the command line")
    parser.add_argument("--custom_name", type=str, default=None, help="Custom save name")
    
    parser.add_argument("--seeds", type=int, default=None, help="Run specific seeds")
    parser.add_argument("--seeds_num", type=int, default=20, help="20 -> all, 5 -> small")
    parser.add_argument("--seeds_start", type=int, default=0)
    parser.add_argument("--learning_rate", type=float, default=[0.1], nargs="*", help="Learning Rate")
    parser.add_argument("--prediction_samples_num", type=int, default=10, help="Number of samples used to get predictions")
    parser.add_argument("--bias", type=lambda x: (str(x).lower() == 'true'), default=False, help="Use bias")
    parser.add_argument("--initial_mean", type=float, default=0, help="Initial mean of priors")#
    parser.add_argument("--initial_var", type=float, default=1, help="Initial variance of priors")
    parser.add_argument("--stopping_var", type=str, default="acc", help="acc, ood, ood_wo_network")
    # Score
    parser.add_argument("--pred_score", type=str, default="softmax", help="P-PPNP/MixedPPNP: How is the prediction score calculated ['distribution', 'softmax']")

    parser.add_argument("--load", type=str, default=None, help='Load model from saved_models. Model needs to be of same class. (only specify date: "2021-11-08 15:24:00")')
    parser.add_argument("--skip_training", type=lambda x: (str(x).lower() == 'true'), default=False, help="Skip training")
    parser.add_argument("--train_mean", type=lambda x: (str(x).lower() == 'true'), default=True, help="Train mean")
    parser.add_argument("--train_var", type=lambda x: (str(x).lower() == 'true'), default=True, help="Train var")
    parser.add_argument("--network_effects", type=lambda x: (str(x).lower() == 'true'), default=True, help="Enable network effects")
    parser.add_argument("--weight_prior", type=str, default="normal", help="normal, laplace or uniform, none")
    parser.add_argument("--max_epochs", type=int, default=[1000], nargs="*", help="number of max epochs per run")
    parser.add_argument("--training_samples_num", type=int, default=1, help="Number of samples used for training step")
    parser.add_argument("--mode", type=str, default="P-PPNP", help="PPNP or P-PPNP or Mixed-PPNP")
    parser.add_argument("--reg_lambda", type=float, default=[0], nargs="*", help="Regularization")
    parser.add_argument("--drop_prob", type=float, default=0, help='Dropout probability')
    parser.add_argument("--edge_drop_prob", type=float, default=0, help='Dropout probability')
    parser.add_argument("--iters_per_seed", type=int, default=5, help="Iterations per seed")
    
    parser.add_argument("--train_without_network_effects", type=lambda x: (str(x).lower() == 'true'), default=False)
    parser.add_argument("--guide_init_scale", type=float, default=0.1)
    parser.add_argument("--optim", type=str, default="clipped_adam", help="adam, clipped_adam")
    parser.add_argument("--clip_norm", type=float, default=[10.0], nargs="*", help="for clipped_adam")
    parser.add_argument("--lr_decay", type=float, default=[1.0], nargs="*", help="for clipped_adam")
    parser.add_argument("--hidden_units", type=int, nargs="*", default=[64])
    
    # GAT
    parser.add_argument("--gat_heads_per_layer", type=int, default=[3], nargs="*")
    parser.add_argument("--gat_add_skip_connections", type=lambda x: (str(x).lower() == 'true'), default=True)

    # OOD 
    parser.add_argument("--ood", type=str, default="none", help="'none', 'loc': leave out classes, perturb")
    parser.add_argument("--ood_eval_during_training", type=lambda x: (str(x).lower() == 'true'), default=True, help="Enables ood evaluation during training")
    # LOC
    parser.add_argument("--loc_classes", nargs="+", default=None, help="List of left out classes [used first if defined]")
    parser.add_argument("--loc_num_classes", type=int, default=None, help="Number of left of classes [used if loc_classes not defined]")
    parser.add_argument("--loc_frac", type=float, default=0.45, help="Fraction of left out classes [used if loc_classes and loc_num_classes not defined]")
    parser.add_argument("--loc_last_classes", type=lambda x: (str(x).lower() == 'true'), default=True, help="Should random classes or last classes be picked [used if loc_classes not specified]")
    parser.add_argument("--remove_loc_classes", type=lambda x: (str(x).lower() == 'true'), default=False, help="Only if loc_last_classes is used: Removes classes from predictable classes")
    parser.add_argument("--loc_remove_edges", type=lambda x: (str(x).lower() == 'true'), default=False)
    # Perturb
    parser.add_argument("--perturb_train",  type=lambda x: (str(x).lower() == 'true'), default=False, help="Whether or not to perturb training set")
    parser.add_argument("--perturb_noise_scale", type=float, default=1.0, help="scale-factor for feature perturbations")
    parser.add_argument("--perturb_custom_p", type=float, default=0.5, help="custom prob for bernoulli perturbation")
    parser.add_argument("--perturb_budget", type=float, default=0.1, help="fraction of perturbed nodes in the graph")
    parser.add_argument("--perturb_mode", type=str, default="normal", help="normal or bernoulli_0.5, shuffle, zeros, ones")

    parser.add_argument("--uncertainty", type=str, default="entropy_per_sample_mean")
    parser.add_argument("--patience", nargs="*", type=int, default=[100])
    parser.add_argument("--use_early_stopping", nargs="*", type=lambda x: (str(x).lower() == 'true'), default=[True])
    parser.add_argument("--optimizers", nargs="*", type=str, default=["training"])
    parser.add_argument("--number_of_eras", type=int, default=1)

    # Combined Uncertainty
    parser.add_argument("--network_mode_uncertainty_aggregation_network_lambda", type=float, default=0.5, help="lambda in [0, 1]")
    parser.add_argument("--network_mode_uncertainty_aggregation_normalize", type=lambda x: (str(x).lower() == 'true'), default=False)
    parser.add_argument("--network_mode_uncertainty_aggregation", type=str, default="min", help="mean, max, min")

    parser.add_argument("--wandb_logging_during_training", type=lambda x: (str(x).lower() == 'true'), default=True)

    # Normalization
    parser.add_argument("--binary_attributes", type=lambda x: (str(x).lower() == 'true'), default=False)
    parser.add_argument("--normalize_attributes", type=str, default='default', help='no, default or div_by_sum')
    
    parser.add_argument("--disable_dropout_on_input", type=lambda x: (str(x).lower() == 'true'), default=False, help="On GCN or PPNP")
    parser.add_argument("--vectorize", type=lambda x: (str(x).lower() == 'true'), default=True)

    args = parser.parse_args()
    
    return args

def overwrite_with_seml_args(args, seml_args):
    if seml_args:
        args.__dict__.update(seml_args)

def overwrite_with_config_args(args):
    # Load config from yaml if available
    if args.config:
        for config_path in args.config:
            with open(os.getcwd() + config_path, 'r') as stream:
                parsed_yaml = yaml.safe_load(stream)
                args.__dict__.update(parsed_yaml)
    