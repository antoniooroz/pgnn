from enum import Enum
from pgnn.utils import StopVariable

class TrainingConfiguration():
    def __init__(self):
        self.learning_rate: float = 0.1
        self.reg_lambda: float = 0.0
        self.early_stopping_variables: list[StopVariable] = [StopVariable.LOSS]
        self.early_stopping: bool = True
        self.skip_training: bool = False
        self.max_epochs: list[int] = 1000
        self.patience: list[int] = [100]
        self.drop_prob: float = 0.0
        self.train_without_network_effects: bool = False
        self.optimizer: OptimizerType = OptimizerType.CLIPPED_ADAM 
        self.optimizer_clip_norm: float = 10.0 # CLIPPED_ADAM
        self.optimizer_learning_rate_decay: float = 1.0 # CLIPPED_ADAM
        self.optimizers: list[str] = ['training']
        self.eras: int = 1
        self.wandb_logging_during_training: bool = True
        
        # OOD
        self.ood_eval_during_training: bool = False
        
        # Edge Dropout
        self.edge_drop_prob: float = 0.0
        
class OptimizerType(Enum):
    CLIPPED_ADAM = 'clipped_adam'
    ADAM = 'adam'