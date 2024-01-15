from transformers import TrainerCallback, TrainerState, TrainerControl
import numpy as np


class EarlyStoppingCallback(TrainerCallback):
    def __init__(self, early_stopping_patience=5):
        self.early_stopping_patience = early_stopping_patience
        self.best_metric = None
        self.no_improvement = 0

    def on_evaluate(self, args, state: TrainerState, control: TrainerControl, metrics, **kwargs):
        current_metric = metrics.get("eval_accuracy")  # Replace with your evaluation metric key

        if self.best_metric is None or current_metric > self.best_metric:
            self.best_metric = current_metric
            self.no_improvement = 0
        else:
            self.no_improvement += 1

        if self.no_improvement >= self.early_stopping_patience:
            print("No improvement on metric for 5 evaluations. Stopping training.")
            control.should_training_stop = True