from .log_weight import LogWeight
import wandb
import pgnn.utils.stat_helpers as stat_helpers

class Logger():
    def __init__(self, model_name, mode):
        self.wandb_run = wandb.init(project="PGNN", name = mode + ' [' + model_name + ']', reinit=True)
        self.iterations:list[LogStep] = []
        
    def newIteration(self, seed, iteration):
        self.iterations.append(LogIteration(seed=seed, iteration=iteration))
        
    def logStep(self, phase, logs, ood, weights):
        self.iterations[-1].logStep(phase, logs, ood, weights)
        
    def logEval(self, phase, logs, ood, weights):
        self.iterations[-1].logEval(phase, logs, ood, weights)
        
    def logAdditionalStats(self, stats):
        self.iterations[-1].logAdditionalStats(stats)
        
    def finish(self):
        results_table = self.createResultsTable()
        
        logs = {}
        for column in results_table.columns:
            logs = {**logs, **stat_helpers.get_stats_for_column(results_table, column, column)}
        
        wandb.log({
            'results_table' : results_table,
            **logs
        })
        
        self.wandb_run.finish()
        
    def finishIteration(self):
        pass
    
    def createResultsTable(self) -> wandb.Table:
        all_results = list(map(lambda x: x.getEvaluationResults(), self.iterations))
        
        columns = set()
        for results in all_results:
            columns.update(results.keys())
            
        columns = list(columns)
        data = []
        for results in all_results:
            data_for_iter = []
            for column in columns:
                data_for_iter.append(results[column])
            data.append(data_for_iter)
            
        results_table = wandb.Table(data=data, columns=columns)
        
        return results_table
        
        
    def watch(self, model):
        wandb.watch(model)
        
    

class LogIteration():
    def __init__(self, seed, iteration):
        self.seed = seed
        self.iteration = iteration
        self.step = 0
        
        self.log_training = {
            'train': [],
            'stopping': []
        }
        self.log_evaluation = {
            'train': None,
            'stopping': None,
            'valtest': None
        }
        self.additionalStats=None
        
    def logStep(self, phase, logs, ood, weights):
        logStep = LogStep(self.step, 'train', phase, logs, ood, weights)
        self.log_training[phase].append(logStep)
        if wandb.config.wandb_logging_during_training:
            logStep.wandbLog()
        self.step+=1
        
    def logEval(self, phase, logs, ood, weights):
        self.log_evaluation[phase] = LogStep(0, 'eval', phase, logs, ood, weights)
        
    def logAdditionalStats(self, stats):
        self.additionalStats = stats
        
    def getEvaluationResults(self):
        data = {'seed': self.seed, 'iteration': self.iteration, 'steps': self.step}
        for logStep in self.log_evaluation.values():
            data.update(logStep.getResults())
            
        return data
            
        
class LogStep():
    def __init__(self, step, mode, phase, logs, ood = None, weights: dict[str, LogWeight] = None):
        self.mode = mode
        self.phase = phase
        self.log_prefix = self.mode + '_' + self.phase
        
        self.step = step
        self.logs = logs
        self.ood = ood if ood else {}
        self.weights = weights
        
    def getResults(self):
        results = {}
        
        for key, val in self.logs.items():
            results[self.log_prefix + '/' + key] = val
        
        for key, val in self.ood.items():
            results['ood_' + self.log_prefix + '/' + key] = val
            
        return results
         
        
    def wandbLog(self):        
        wandb.log(self.getResults())
  

    

    