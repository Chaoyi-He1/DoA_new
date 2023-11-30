import torch
import torch.nn as nn
from util.misc import *
from typing import Iterable


def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, max_norm: float = 0.01,
                    scaler=None):
    
    model.train()
    criterion.train()
    
    metric_logger = MetricLogger(delimiter="; ")