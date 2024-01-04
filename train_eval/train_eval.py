import torch
import torch.nn as nn
from util.misc import *
from typing import Iterable
from pycocotools.coco import COCO
from .coco_eval import CocoEvaluator


def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, max_norm: float = 0.01,
                    scaler=None):
    
    model.train()
    criterion.train()
    
    metric_logger = MetricLogger(delimiter="; ")
    metric_logger.add_meter('loss', SmoothedValue(window_size=10, fmt='{value:.6f}'))
    metric_logger.add_meter('lr', SmoothedValue(window_size=10, fmt='{value:.6f}'))
    
    header = 'Epoch: [{}]'.format(epoch)
    
    for samples, targets in metric_logger.log_every(data_loader, 50, header):
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        
        with torch.cuda.amp.autocast(enabled=scaler is not None):
            outputs = model(samples)
            
            loss_dict = criterion(outputs, targets)
            weight_dict = criterion.weight_dict
            
            weighted_losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)
            
        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = reduce_dict(loss_dict)
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}
        
        # backward
        optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(weighted_losses).backward()
        else:
            weighted_losses.backward()

        if max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            
        if scaler is not None:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        
        metric_logger.update(loss=weighted_losses.item(), **loss_dict_reduced_unscaled)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
    
    # Gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(model: torch.nn.Module, criterion: torch.nn.Module,
             data_loader: Iterable, device: torch.device, 
             postprocessors: dict, base_ds: COCO,
             scaler=None):
    
    model.eval()
    criterion.eval()
    
    metric_logger = MetricLogger(delimiter="; ")
    metric_logger.add_meter('loss', SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Test:'
    
    iou_types = tuple(k for k in ('segm', 'bbox') if k in postprocessors.keys())
    coco_evaluator = CocoEvaluator(base_ds, iou_types)
    
    for samples, targets in metric_logger.log_every(data_loader, 10, header):
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        
        with torch.cuda.amp.autocast(enabled=scaler is not None):
            outputs = model(samples)
            
            loss_dict = criterion(outputs, targets)
            weight_dict = criterion.weight_dict
            
            weighted_losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)
        
        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = reduce_dict(loss_dict)
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}
        metric_logger.update(loss=weighted_losses.item(),
                             **loss_dict_reduced_unscaled)
        metric_logger.update(class_error=loss_dict['class_error'].item())
        
        # For coco evaluation
        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
        results = postprocessors['bbox'](outputs, orig_target_sizes)
        res = {target['image_id'].item(): output for target, output in zip(targets, results)}
        if coco_evaluator is not None:
            coco_evaluator.update(res)
    
    # Gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    coco_evaluator.synchronize_between_processes()
    coco_evaluator.accumulate()
    coco_evaluator.summarize()

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}, coco_evaluator


@torch.no_grad()
def evaluate_cnn_test(model: torch.nn.Module, criterion: torch.nn.Module,
                      data_loader: Iterable, device: torch.device,
                      scaler=None):
    
    model.eval()
    criterion.eval()
    
    metric_logger = MetricLogger(delimiter="; ")
    metric_logger.add_meter('loss', SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Test:'
    
    for samples, targets in metric_logger.log_every(data_loader, 10, header):
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        
        with torch.cuda.amp.autocast(enabled=scaler is not None):
            outputs = model(samples)
            
            loss_dict = criterion(outputs, targets)
            weight_dict = criterion.weight_dict
            
            weighted_losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)
        
        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = reduce_dict(loss_dict)
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}
        metric_logger.update(loss=weighted_losses.item(),
                             **loss_dict_reduced_unscaled)
    
    # Gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}