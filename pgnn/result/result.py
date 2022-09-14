from torch import Tensor

class Result():
    def __init__(self):
        pass
        
        
class ModelOutputs():
    def __init__(self,
            softmax_scores: Tensor = None,
            predicted_class: Tensor = None,
            epistemic_uncertainties: Tensor = None,
            aleatoric_uncertainties: Tensor = None
    ):
        self.softmax_scores: Tensor = softmax_scores
        self.predicted_class: Tensor = predicted_class
        self.epistemic_uncertainties = epistemic_uncertainties
        self.aleatoric_uncertainties = aleatoric_uncertainties
        