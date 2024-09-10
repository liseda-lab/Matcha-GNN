import numpy, torch
from typing import Union

class Evaluator:
    def __init__(self, main_metric: str, secondary_metrics: Union[list,str]):
        #__init__(self, main_metric: str, secondary_metrics: list | str):

        self.main_metric = main_metric
        self.secondary_metrics = secondary_metrics
        self.results = {}


    def evaluate(self, y_pred, y_true):
        """
        Evaluates the model's performance on the given data.
        Both y_pred and y_true should be of the same type and length.

        y_pred: torch.Tensor | numpy.ndarray
        y_true: torch.Tensor | numpy.ndarray
        """
        self.assess_data_type(y_pred, y_true)
        return self.run_eval()


    def assess_data_type(self, y_pred, y_true):
        """
        Asserts that the data types of y_pred and y_true are either torch.Tensor or numpy.ndarray and that 
        they have the same length.

        y_pred: torch.Tensor | numpy.ndarray
        y_true: torch.Tensor | numpy.ndarray
        """
        assert type(y_pred) == torch.Tensor or numpy.ndarray
        assert len(y_pred) == len(y_true)
        self.y_pred = y_pred
        self.y_true = y_true


    def run_eval(self):
        """
        Evaluates the model's performance on the given data using the specified main and secondary metrics.
        Returns: dict
        """
        for metric in self.secondary_metrics:
            if metric == 'accuracy': self.results[metric] = self.eval_accuracy()
            elif metric == 'f1' or metric == 'precision' or metric == 'recall': self.results[metric] = self.eval_f1()[metric]
            else: raise ValueError(f"Metric {metric} not supported.")
        if self.main_metric == 'accuracy': self.results[self.main_metric] = self.eval_accuracy()
        elif self.main_metric == 'f1' or self.main_metric == 'precision' or self.main_metric == 'recall':
            self.results[self.main_metric] = self.eval_f1()[self.main_metric]
        else: raise ValueError(f"Metric {self.main_metric} not supported.")


    def eval_accuracy(self):
        """
        Evaluates the model's accuracy on the given data.
        Returns: float
        """
        y_pred = [1 if y > 0.5 else 0 for y in self.y_pred].numpy()
        y_true = self.y_true.numpy()
        accuracy = sum([1 for i in range(len(y_pred)) if y_pred[i] == y_true[i]]) / len(y_pred)
        return accuracy


    def eval_f1(self):
        """
        Evaluates the model's F1 score on the given data.
        Returns: dict {"f1":float, "precision":float, "recall":float}
        """      
        y_pred = [1 if y > 0.5 else 0 for y in self.y_pred].numpy()
        y_true = self.y_true.numpy()
        tp = sum([1 for i in range(len(y_pred)) if y_pred[i] == 1 and y_true[i] == 1])
        fp = sum([1 for i in range(len(y_pred)) if y_pred[i] == 1 and y_true[i] == 0])
        fn = sum([1 for i in range(len(y_pred)) if y_pred[i] == 0 and y_true[i] == 1])
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        f1 = 2 * (precision * recall) / (precision + recall)
        return {"f1":f1, 
                "precision":precision, 
                "recall":recall}
