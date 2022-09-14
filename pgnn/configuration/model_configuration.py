from enum import Enum, auto


class ModelConfiguration():
    def __init__(self):
        # Bayesian
        self.samples_prediction: int = 10
        self.samples_training: int = 3
        self.initial_mean: float = 0
        self.initial_var: float = 0
        self.train_mean: bool = True
        self.train_var: bool = True
        self.pred_score: str = 'softmax'
        self.weight_prior: str = 'normal'
        self.guide_init_scale: float = 0.1
        self.uncertainty_estimation: str = 'entropy_per_sample_mean'
        self.vectorize: bool = True
        self.network_mode_uncertainty_aggregation_network_lambda: float = 0.5
        self.network_mode_uncertainty_aggregation_normalize: bool = False
        self.network_mode_uncertainty_aggregation: str = 'min' # 'min', 'max', 'mean'
        
        # Default
        self.model: ModelType = ModelType.PPNP
        self.bias: bool = False
        self.network_effects: bool = True
        self.hidden_layer_size: list[int] = [64]
        
        # GAT
        self.gat_hidden_layers_heads: list[int] = [3]
        self.gat_skip_connections: bool = True 
        
        # PPNP 
        self.disable_dropout_on_input: bool = False
        
        # GPN
        self.binary_attributes: bool = False
        self.normalize_attributes: str = 'no' # 'no', 'default', 'div_by_sum'
        
        
        
class ModelType(Enum):
    PPNP = auto()
    P_PPNP = auto()
    # TODO ...
    