from enum import Enum

class ExperimentConfiguration():
    def __init__(self):
        self.experiment_mode: ExperimentMode = ExperimentMode.DEVELOPMENT
        self.dataset : Dataset = Dataset.CITESEER
        self.seeds: Seeds = None # Afterwards
        self.iterations = 5 # Iterations per seed
        
        # OOD
        self.ood: OOD = OOD.NONE 
        self.ood_loc_classes: list[int] = None
        self.ood_loc_num_classes: int = None
        self.ood_loc_frac: float = 0.45
        self.ood_loc_remove_classes: bool = False
        self.ood_loc_remove_edges: bool = True
        self.ood_perturb_train: bool = False
        self.ood_perturb_noise_scale: float = 1.0
        self.ood_perturb_bernoulli_probability: float = 0.5
        self.ood_perturb_budget: float = 0.1
        self.ood_perturb_mode: OODPerturbMode = OODPerturbMode.BERNOULLI
        
class ExperimentMode(Enum):
    DEVELOPMENT = 'development'
    TEST = 'test'
    
class Dataset(Enum):
    CORA_ML = 'cora_ml'
    CITESEER = 'citeseer'

class OOD(Enum):
    NONE = 'none'
    LOC = 'loc'
    PERTURB = 'perturb'
    
class OODPerturbMode(Enum):
    NORMAL = 'normal'
    BERNOULLI = 'bernoulli'
    SHUFFLE = 'shuffle'
    ZEROS = 'zeros'
    ONES = 'ones'

class Seeds():
    def __init__(self, 
            experiment_mode: ExperimentMode = ExperimentMode.DEVELOPMENT,
            specific_seeds: list[int] = None,
            start: int = 0,
            end: int = 20):
        self._specific_seeds = specific_seeds
        self._start = start
        self._end = end
        self.seeds = self._get_seeds(experiment_mode=experiment_mode)
        
    def _get_seeds(self, experiment_mode):
        test_seeds = [
            2144199730,  794209841, 2985733717, 2282690970, 1901557222,
            2009332812, 2266730407,  635625077, 3538425002,  960893189,
            497096336, 3940842554, 3594628340,  948012117, 3305901371,
            3644534211, 2297033685, 4092258879, 2590091101, 1694925034
        ]
        development_seeds = [
            2413340114, 3258769933, 1789234713, 2222151463, 2813247115,
            1920426428, 4272044734, 2092442742, 841404887, 2188879532,
            646784207, 1633698412, 2256863076,  374355442,  289680769,
            4281139389, 4263036964,  900418539,  119332950, 1628837138
        ]
        
        if self._specific_seeds:
            self.seeds = self._specific_seeds
        else:
            if experiment_mode==ExperimentMode.TEST:
                self.seeds = test_seeds
            else:
                self.seeds = development_seeds
            self.seeds = self.seeds[self._start:self._end]
    