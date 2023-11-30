import argparse
import datetime
import glob
import json
import math
import os
import random
import time
import pickle
from pathlib import Path
import tempfile

import yaml
import torch
import torch.distributed as dist
import numpy as np
from util.misc import torch_distributed_zero_first
from util.distributed_util import Custom_DistributedSampler
from torch.utils.tensorboard import SummaryWriter
import torch.optim.lr_scheduler as lr_scheduler
import torch.multiprocessing

import util.misc as utils
from datasets.Dataset import *
from models.detection import *
from train_eval.train_eval import *
from util.coco_util import *


torch.multiprocessing.set_sharing_strategy('file_system')


def get_args_parser():
    parser = argparse.ArgumentParser('Set transformer detector', add_help=False)
    parser.add_argument('--device', default='cuda', help='device id (i.e. 0 or 0,1 or cpu)')
    parser.add_argument('--name', default='', help='renames results.txt to results_name.txt if supplied')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--eval', action='store_true', help='only evaluate model on validation set')

    # Model parameters
    parser.add_argument('--resume', type=str, default='weights/contrast/model_0395.pth', help="initial weights path")  # weights/model_940.pth
    parser.add_argument('--hpy', type=str, default='cfg/cfg.yaml', help="hyper parameters path")
    parser.add_argument('--positional-embedding', default='sine', choices=('sine', 'learned'),
                        help="type of positional embedding to use on top of the image features")
    parser.add_argument('--sync-bn', action='store_true', help='enabling apex sync BN.')
    parser.add_argument('--freeze-encoder', default=False, help="freeze the encoder")
    parser.add_argument('--save-best', action='store_true', help="save best model")

    # Optimization parameters
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--lrf', default=0.001, type=float)
    parser.add_argument('--weight_decay', default=0.0, type=float)
    parser.add_argument('--epochs', default=1000, type=int)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N', help='start epoch')

    # dataset parameters
    parser.add_argument('--train-path', default='/data/share/arya/DOA/train_2/', help='train dataset path')
    parser.add_argument('--val-path', default='/data/share/arya/DOA/val_2/', help='val dataset path')
    parser.add_argument('--cache-data', default=False, type=bool, help='cache data for faster training')
    parser.add_argument('--output-dir', default='weights/', help='path where to save, empty for no saving')

    # distributed training parameters
    parser.add_argument('--world_size', default=2, type=int, help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--amp', action='store_false', help='use mixed precision')
    
    return parser


def main(args):
    utils.init_distributed_mode(args)
    if args.amp:
        assert torch.backends.cudnn.enabled, \
            "NVIDIA Apex extension is not available. Please check environment and/or dependencies."
        assert torch.backends.cudnn.version() >= 7603, \
            "NVIDIA Apex extension is outdated. Please update Apex extension."
    if args.rank in [-1, 0]:
        print(args)

        print('Start Tensorboard with "tensorboard --logdir=runs", view at http://localhost:6006/')
        tb_writer = SummaryWriter(comment=args.name)
    
    device = torch.device(args.device)
    if "cuda" not in args.device:
        raise EnvironmentError("CUDA is not available, please check your environment settings.")
    
    best = args.output_dir + os.sep + "best.pt"
    output_dir = Path(args.output_dir)
    results_file = "results{}.txt".format(datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    
    # Remove previous results
    if args.rank in [-1, 0]:
        for f in glob.glob(results_file) + glob.glob("tmp.pk"):
            os.remove(f)
    
    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    # load hyper parameters
    with open(args.hpy) as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    
    # dataset generate
    print("DoA dataset generating...")
    dataset_train = LoadDataAndLabels(folder_path=args.train_path, cache=args.cache_data, train=True,
                                      data_size=cfg['data_size'], rank=args.rank)
    dataset_val = LoadDataAndLabels(folder_path=args.val_path, cache=args.cache_data, train=False,
                                    data_size=cfg['data_size'], rank=args.rank)
    print("DoA dataset generated.")
    
    if args.distributed:
        sampler_train = torch.utils.data.distributed.DistributedSampler(dataset_train)
        sampler_val = torch.utils.data.distributed.DistributedSampler(dataset_val, shuffle=False)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)
    
    batch_sampler_train = torch.utils.data.BatchSampler(sampler_train, args.batch_size, drop_last=True)
    batch_sampler_val = torch.utils.data.BatchSampler(sampler_val, args.batch_size, drop_last=True)
    
    # dataloader
    print("Dataloader generating...")
    nw = min([os.cpu_count(), args.batch_size if args.batch_size > 1 else 0, 8])  # number of wor-kers
    if args.rank in [-1, 0]:
        print('Using %g dataloader workers' % nw)
    data_loader_train = torch.utils.data.DataLoader(dataset_train, batch_sampler=batch_sampler_train, 
                                                    collate_fn=dataset_train.collate_fn, num_workers=nw)
    data_loader_val = torch.utils.data.DataLoader(dataset_val, batch_sampler=batch_sampler_val,
                                                  collate_fn=dataset_val.collate_fn, num_workers=nw)
    
    base_ds = get_coco_api_from_dataset(dataset_val)
    
    # model
    print("Model generating...")
    model, criterion, postprocessors = build(cfg)
    model.to(device)
    if args.rank in [-1, 0] and tb_writer:
        tb_writer.add_graph(model, torch.rand((1, 12, 512, 512), 
                                              device=device, dtype=torch.float), use_strict_trace=False)
    
     # load previous model if resume training
    start_epoch = 0
    scaler = torch.cuda.amp.GradScaler() if args.amp else None
    
    if args.resume.endswith(".pth"):
        print("Resuming training from %s" % args.resume)
        ckpt = torch.load(args.resume, map_location='cpu')
        try:
            ckpt["model"] = {k: v for k, v in ckpt["model"].items()
                             if model.state_dict()[k].numel() == v.numel()}
            model.load_state_dict(ckpt["model"], strict=False)
        except KeyError as e:
            s = "%s is not compatible with %s. Specify --weights '' or specify a --cfg compatible with %s. " \
                "See https://github.com/ultralytics/yolov3/issues/657" % (args.weights, args.hyp, args.weights)
            raise KeyError(s) from e
        
        if args.rank in [-1, 0]:
            # load results
            if ckpt.get("training_results") is not None:
                with open(results_file, "w") as file:
                    file.write(ckpt["training_results"])  # write results.txt
        
        # epochs
        start_epoch = ckpt["epoch"] + 1
        if args.epochs < start_epoch:
            print('%s has been trained for %g epochs. Fine-tuning for %g additional epochs.' %
                  (args.weights, ckpt['epoch'], args.epochs))
            args.epochs += ckpt['epoch']  # finetune additional epochs
        if args.amp and "scaler" in ckpt:
            scaler.load_state_dict(ckpt["scaler"])
        del ckpt
        print("Loading model from: ", args.weights, "finished.")
    
    # freeze backbone if args.freeze_encoder is true
    if args.freeze_encoder:
        for name, param in model.named_parameters():
            if "backbone" in name:
                param.requires_grad = False
        print("Encoder frozen.")
        
    # synchronize batch norm layers if args.sync_bn is true
    if args.sync_bn:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).to(device)
        print("Sync BatchNorm layers.")
    
    # distributed model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module
    
    # model info
    params_to_optimize = []
    n_parameters, layers = 0, 0
    for p in model.parameters():
        n_parameters += p.numel()
        layers += 1
        if p.requires_grad:
            params_to_optimize.append(p)
    print('Model Summary: %g layers, %g parameters' % (layers, n_parameters))

    # learning rate scheduler setting
    # After using DDP, the gradients on each device will be averaged, so the learning rate needs to be enlarged
    args.lr *= max(1., args.world_size * args.batch_size / 64)
    optimizer = torch.optim.Adam(params_to_optimize, lr=args.lr, weight_decay=args.weight_decay)
    lf = lambda x: ((1 + math.cos(x * math.pi / args.epochs)) / 2) * (1 - args.lrf) + args.lrf  # cosine
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
    scheduler.last_epoch = start_epoch  # do not move
    
    # start training
    if args.rank in [-1, 0]:
        print("starting traning for %g epochs..." % args.epochs)
        print('Using %g dataloader workers' % nw)
    
    if args.eval:
        test_stats, coco_evaluator = evaluate(model, criterion, postprocessors,
                                              data_loader_val, base_ds, device, args.output_dir)
        if args.output_dir:
            utils.save_on_master(coco_evaluator.coco_eval["bbox"].eval, output_dir / "eval.pth")
        return

    print("Start training...")
    best_loss = float('inf')
    start_time = time.time()
    for epoch in range(start_epoch, args.epochs + start_epoch):
        if args.distributed:
            sampler_train.set_epoch(epoch)
            
        train_stats = train_one_epoch(model=model, criterion=criterion, 
                                      data_loader=data_loader_train, device=device, 
                                      optimizer=optimizer, epoch=epoch, 
                                      scaler=scaler)
        scheduler.step()
        
        test_stats, coco_evaluator = evaluate(model=model, data_loader=data_loader_val, 
                                              criterion=criterion, device=device, 
                                              postprocessors=postprocessors, 
                                              base_ds=base_ds, scaler=scaler)
        
        # write results
        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     **{f'test_{k}': v for k, v in test_stats.items()},
                     'epoch': epoch,
                     'n_parameters': n_parameters}
        
        if args.output_dir and utils.is_main_process():
            with (Path(results_file)).open("a") as f:
                f.write(json.dumps(log_stats) + "\n")
        
        # write tensorboard
        if utils.is_main_process():
            if tb_writer:
                items = {
                    **{f'train_{k}': v for k, v in train_stats.items()},
                    **{f'test_{k}': v for k, v in test_stats.items()},
                }
                for k, v in items.items():
                    tb_writer.add_scalar(k, v, epoch)
        
        # save model
        if args.save_best:
            # save best model
            if test_stats["loss"] < best_loss:
                best_loss = test_stats["loss"]

                utils.save_on_master({
                    "epoch": epoch,
                    "model": model_without_ddp.state_dict() if args.distributed else model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scaler": scaler.state_dict() if scaler is not None else None,
                    "lr_scheduler": scheduler.state_dict(),
                }, best)
        else:
            # save latest model
            digits = len(str(args.epochs))
            utils.save_on_master({
                "epoch": epoch,
                "model": model_without_ddp.state_dict() if args.distributed else model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scaler": scaler.state_dict() if scaler is not None else None,
                "lr_scheduler": scheduler.state_dict(),
            }, os.path.join(args.output_dir, 'model_{}.pth'.format(str(epoch).zfill(digits))))
    
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))
    

if __name__ == '__main__':
    parser = get_args_parser()
    args = parser.parse_args()
    # Check if files exist
    args.hpy = check_file(args.hpy)
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)