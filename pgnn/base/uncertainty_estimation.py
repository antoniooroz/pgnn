import torch

uncertainty_metrics = {}

def uncertainty_metric(func):
    """
        Adapted from https://stackoverflow.com/a/54075852/17134758
        Last visited: 06.07.2022
    """
    uncertainty_metrics[func.__name__] = func
    return func

def get_uncertainty(config, uncertainty_metric, **data):
    uncertainty_metric_fn = uncertainty_metrics.get(uncertainty_metric, None)
    if uncertainty_metric_fn:
        return uncertainty_metric_fn(data)
    else:
        raise NotImplementedError() 
    
@uncertainty_metric
def probability(data):
    probs_mean = data["probs_mean"]
    return probs_mean.max(-1).values.squeeze(0)

@uncertainty_metric
def min_probability(data):
    probs_all, preds = data["probs_all"], data["preds"]
    min_probs = probs_all.min(0).values.squeeze(0)
    return min_probs.gather(dim=-1, index=preds.unsqueeze(-1))

@uncertainty_metric
def max_probability(data):
    probs_all, preds = data["probs_all"], data["preds"]
    max_probs = probs_all.max(0).values.squeeze(0)
    return max_probs.gather(dim=-1, index=preds.unsqueeze(-1))

@uncertainty_metric
def variance_in_prediction_probability(data):
    probs_all, probs_mean, preds = data["probs_all"], data["probs_mean"], data["preds"]
    
    N = probs_all.shape[0]
    probs_all = probs_all.squeeze(1)
    preds = preds.unsqueeze(-1)
    probs_for_class = probs_all.gather(dim=-1, index=preds.unsqueeze(0).repeat_interleave(N, dim=0)).squeeze(-1)
    variance = probs_for_class.var(dim=0)
    return -variance

@uncertainty_metric
def variance_between_samples_per_class_summed(data):
    probs_all, probs_mean, preds = data["probs_all"], data["probs_mean"], data["preds"]
    
    probs_all = probs_all.squeeze(1)
    variance = probs_all.var(dim=0)
    variance = variance.sum(dim=-1)
    return -variance

@uncertainty_metric
def variance_logits_between_samples_per_class_summed(data):
    logits = data["logits"]
    
    logits = logits.squeeze(1)
    variance = logits.var(dim=0)
    variance = variance.sum(dim=-1)
    return -variance

@uncertainty_metric
def logits(data):
    logits = data["logits"]

    logits = logits.squeeze(1)
    return logits.mean(dim=0).mean(dim=-1).exp()

@uncertainty_metric
def variance_in_prediction_probability_normalized(data):
    probs_all, probs_mean, preds = data["probs_all"], data["probs_mean"], data["preds"]
    
    N = probs_all.shape[0]
    probs_all = probs_all.squeeze(1)
    preds = preds.unsqueeze(-1)
    probs_for_class = probs_all.gather(dim=-1, index=preds.unsqueeze(0).repeat_interleave(N, dim=0)).squeeze(-1)
    probs_mean_for_class = probs_mean.gather(dim=-1, index=preds).squeeze(-1)
    variance = probs_for_class.var(dim=0)
    return -variance / (probs_mean_for_class + 1e-6)

@uncertainty_metric
def mean_variance_in_all_probabilities(data):
    probs_all, probs_mean = data["probs_all"], data["probs_mean"]
    
    probs_all = probs_all.squeeze(1)
    variance = probs_all.var(dim=0)
    return -variance.mean(dim=-1)

@uncertainty_metric
def entropy(data):
    probs_mean = data["probs_mean"]
    result = (probs_mean * torch.log2(probs_mean))
    result[probs_mean == 0] = 0 # Fix torch.log2(0) -> nan
    return result.sum(dim=-1)

def _entropy_per_sample(data):
    probs_all = data["probs_all"]
    result = (probs_all * torch.log2(probs_all))
    result[probs_all == 0] = 0 # Fix torch.log2(0) -> nan
    return result.sum(dim=-1)

@uncertainty_metric
def entropy_per_sample_mean(data):
    entropy_per_sample = _entropy_per_sample(data)
    return entropy_per_sample.mean(0).squeeze(0)

@uncertainty_metric
def entropy_per_sample_max(data):
    entropy_per_sample = _entropy_per_sample(data)
    return entropy_per_sample.max(0).values.squeeze(0)

@uncertainty_metric
def entropy_per_sample_min(data):
    entropy_per_sample = _entropy_per_sample(data)
    return entropy_per_sample.min(0).values.squeeze(0)

@uncertainty_metric
def entropy_per_sample_var(data):
    entropy_per_sample = _entropy_per_sample(data)
    N = entropy_per_sample.shape[0]
    
    variance = entropy_per_sample.var(dim=0)
    
    return -variance.squeeze(0)

@uncertainty_metric
def entropy_in_max_probabilities(data):
    probs_all = data["probs_all"]
    
    N = probs_all.shape[0]
    probs_all = probs_all.squeeze(1)
    probs_all_max = probs_all.max(dim=-1).values.unsqueeze(-1)
    probs_all[probs_all < probs_all_max] = 0
    
    probs_only_max_mean = probs_all.mean(dim=0)
    
    result = (probs_only_max_mean * torch.log2(probs_only_max_mean))
    result[probs_only_max_mean == 0] = 0 # Fix torch.log2(0) -> nan
    return result.sum(dim=-1)

@uncertainty_metric
def entropy_per_sample_logits_mean(data):
    if "logits" not in data.keys():
        raise NotImplementedError("Model does not provide logits")
    
    logits = data["logits"].squeeze(1)
    assert len(logits.shape) == 3
    
    logits_exp = torch.exp(logits / logits.max())
    result = (logits_exp * torch.log2(logits_exp))
    result[logits_exp == 0] = 0 # Fix torch.log2(0) -> nan
    return result.sum(dim=-1).mean(0)

@uncertainty_metric
def activation_variance(data):
    if "activation_variance" not in data.keys() or data["activation_variance"] is None:
        raise NotImplementedError("Model does not provide activation variances")

    return -data["activation_variance"].mean(dim=0).squeeze(0)

@uncertainty_metric
def gat_attention_variance(data):
    if "result" not in data.keys() or "edge_scores" not in data["result"] or "edge_indices" not in data["result"]:
        raise NotImplementedError("Model does not provide required data")

    edge_indices = data["result"]["edge_indices"][0].squeeze(0)[0]
    edge_scores = data["result"]["edge_scores"].squeeze(1).squeeze(1)

    variances = edge_scores.var(dim=0)
    variances = variances.mean(dim=-1).unsqueeze(-1).expand(-1, data["idx"].shape[0])
    edge_indices = edge_indices.unsqueeze(-1).expand(-1, data["idx"].shape[0])
    indices_select = data["idx"].unsqueeze(0).expand(edge_indices.shape[0], -1)
    variances = variances.clone()

    variances[edge_indices!=indices_select] = 0

    ones = torch.ones(variances.shape, device=variances.device)
    ones[edge_indices!=indices_select] = 0

    node_variances = (variances.sum(0) / ones.sum(0))

    return -node_variances
    
   
    