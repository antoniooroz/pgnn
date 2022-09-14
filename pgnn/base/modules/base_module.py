from typing import List
from pyro.nn import PyroModule, PyroParam, PyroSample
import torch
from pgnn.logger import LogWeight, LogWeightValue
from pgnn.inits import glorot
from pgnn.utils import get_device
import pyro.distributions as dist

class BaseModule(PyroModule):
    def __init__(self, input_dim, output_dim, config, name=""):
        super().__init__(name)
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.config = config

        self.weight = None
        self.weight_mean = None
        self.weight_var = None

    def pyronize_weights(self):
        # Mean
        self.weight_mean = glorot([self.input_dim, self.output_dim]).to(get_device()) + self.config.initial_mean
        if self.config.train_mean:
            self.weight_mean = PyroParam(self.weight_mean)
        
        # Variance
        self.weight_var = torch.ones([self.input_dim, self.output_dim]).to(get_device()) * self.config.initial_var
        if self.config.train_var:
            self.weight_var = PyroParam(self.weight_var, constraint=dist.constraints.positive)
        
        # Weight
        if self.config.weight_prior == "normal":
            self.weight = PyroSample(lambda self: dist.Normal(self.weight_mean, self.weight_var).to_event(2))
        elif self.config.weight_prior == "laplace":
            self.weight = PyroSample(lambda self: dist.Laplace(self.weight_mean, self.weight_var).to_event(2))
        elif self.config.weight_prior == "none":
            self.weight = self.weight_mean
        else:
            raise NotImplementedError()
    
    def log_weights(self) -> List[LogWeight]:
        if self.weight_mean is not None:
            return [LogWeight(
                name=self._pyro_name,
                mu=LogWeightValue(self.weight_mean),
                sigma=LogWeightValue(self.weight_var)
            )]
        else:
            return [LogWeight(
                name=self._pyro_name,
                mu=LogWeightValue(self.weight)
            )]