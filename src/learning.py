'''
This module contains functions and variables to be used when
training and evaluating a PyTorch model
'''


import math
import sys
import os
import time
import datetime
from collections import defaultdict, deque

import torch
import numpy as np
import wandb

import utils
import evaluators


class SmoothedValue():
    '''
    Track a series of values and provide access to smoothed values over a
    window or the global series average
    (see https://github.com/pytorch/vision/blob/master/references/detection/utils.py)
    '''

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    @property
    def median(self):
        values = torch.tensor(list(self.deque))
        return values.median().item()

    @property
    def avg(self):
        values = torch.tensor(list(self.deque), dtype=torch.float32)
        return values.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value
        )


class MetricLogger(object):
    '''
    Store different series of values, to log specific metrics during training
    '''

    def __init__(self, delimiter="\t", enable_wandb=False):
        self.delimiter = delimiter
        self.enable_wandb = enable_wandb
        self.meters = defaultdict(SmoothedValue)

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor) or isinstance(v, np.generic):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError(
            f"'{type(self).__name__}' object has no attribute '{attr}'"
        )

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(
                "{}: {}".format(name, str(meter))
            )
        return self.delimiter.join(loss_str)

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def wandb_log(self, prefix, time, keys=None):
        logs = dict()
        meters = dict(self.meters, **time)
        keys = meters.keys() if keys is None else keys
        for k in keys:
            if k in meters:
                v = meters[k]
                logs[f'{prefix.lower()}/{k}'] = (
                    v if not isinstance(v, SmoothedValue)
                    else v.value
                )
        wandb.log(logs)

    def log_every(self, iterable, print_freq, header='', prefix=''):
        i = 0
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt='{avg:.4f}')
        data_time = SmoothedValue(fmt='{avg:.4f}')
        space_fmt = ':' + str(len(str(len(iterable)))) + 'd'
        if torch.cuda.is_available():
            log_msg = self.delimiter.join([
                header,
                '[{0' + space_fmt + '}/{1}]',
                'eta: {eta}',
                '{meters}',
                'time: {time}',
                'data: {data}',
                'max mem: {memory:.0f}'
            ])
        else:
            log_msg = self.delimiter.join([
                header,
                '[{0' + space_fmt + '}/{1}]',
                'eta: {eta}',
                '{meters}',
                'time: {time}',
                'data: {data}'
            ])
        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0 or i == len(iterable) - 1:
                if self.enable_wandb:
                    self.wandb_log(prefix, {
                        'data_loading_time': data_time,
                        'iteration_time': iter_time
                    })
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if torch.cuda.is_available():
                    print(
                        log_msg.format(
                            i, len(iterable), eta=eta_string,
                            meters=str(self),
                            time=str(iter_time), data=str(data_time),
                            memory=torch.cuda.max_memory_allocated() / MB
                        )
                    )
                else:
                    print(
                        log_msg.format(
                            i, len(iterable), eta=eta_string,
                            meters=str(self),
                            time=str(iter_time), data=str(data_time)
                        )
                    )
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print(
            '{} Total time: {} ({:.4f} s / it)'.format(
                header, total_time_str, total_time / len(iterable)
            )
        )


def collate_fn(batch):
    '''
    Flatten the given batch, which is a list of lists like the following
    [[(img_1_1, targets_1_1), (img_1_2, targets_1_2), ...],
       [(img_2_1, targets_2_1), ...], ...]

    to be a single list of tuples like the following
    [(img_1_1, targets_1_1), (img_1_2, targets_1_2), ..., (img_2_1, targets_2_1), ...]

    and then zip it, so as to obtain a tuple of tuples like the following
    ((img_1_1, img_1_2, img_2_1, ...), (targets_1_1, targets_1_2, targets_2_1, ...))
    '''
    flattened_batch = list(utils.flatten(batch))
    return tuple(zip(*flattened_batch))


def train_one_epoch(params, model, optimizer, dataloader, epoch):
    '''
    Train the given model for one epoch with the given
    dataloader and parameters
    '''
    # Put the model in training mode
    model.train()

    # Create an instance of the metric logger
    metric_logger = MetricLogger(
        delimiter=" ", enable_wandb=params.generic.wandb.enabled
    )
    metric_logger.add_meter(
        'lr', SmoothedValue(window_size=1, fmt='{value:.6f}')
    )
    dataloader_wrapper = metric_logger.log_every(
        dataloader, params.training.log_interval, header=f'Epoch: [{epoch}]', prefix="train"
    )

    # For each batch of (images, targets) pairs
    for images, targets in dataloader_wrapper:

        # Aggregate losses
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        loss_value = losses.item()

        # Stop if loss is not finite
        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict)
            sys.exit(1)

        # Perform backpropagation
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        # Update metrics
        metric_logger.update(loss=losses, **loss_dict)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    return metric_logger


def save_checkpoint(epoch, params, model, optimizer, lr_scheduler=None):
    '''
    Save the current state of the model, optimizer and learning rate scheduler,
    both locally and on wandb (if available and enabled)
    '''
    state_dict = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'lr_scheduler': (
            lr_scheduler.state_dict()
            if lr_scheduler is not None else dict()
        ),
        'params': params,
        'epoch': epoch
    }
    checkpoint_path = os.path.join(
        params.training.checkpoints.path,
        f'{utils.now()}_{epoch}.pth'
    )
    torch.save(state_dict, checkpoint_path)
    if params.generic.wandb.enabled:
        wandb.save(checkpoint_path)


def training_loop(params, model, optimizer, train_dataloader,
                  val_dataloader, lr_scheduler=None):
    '''
    Standard training loop function: train and evaluate
    after each training epoch
    '''
    # Track execution time
    start_time = time.time()

    # For each epoch
    for epoch in range(1, params.training.epochs + 1):

        # Train for one epoch
        train_one_epoch(params, model, optimizer, train_dataloader, epoch)

        # Decay the learning rate
        if lr_scheduler is not None:
            lr_scheduler.step()

        # Save checkpoints once in a while
        if (params.training.checkpoints.save and
                epoch % params.training.checkpoints.frequency == 0):
            save_checkpoint(
                epoch, params, model, optimizer, lr_scheduler=lr_scheduler
            )

        # Evaluate after every epoch
        evaluate(params, model, val_dataloader)

    # Always save the last checkpoint
    save_checkpoint(
        params.training.epochs, params, model,
        optimizer, lr_scheduler=lr_scheduler
    )

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print(f'Training time {total_time_str}')


@ torch.no_grad()
def evaluate(params, model, dataloader):
    # Put the model in evaluation mode
    model.eval()

    # Create an instance of the metric logger
    metric_logger = MetricLogger(
        delimiter=" ", enable_wandb=params.generic.wandb.enabled
    )
    dataloader_wrapper = metric_logger.log_every(
        dataloader, params.training.log_interval, header=f'Test:', prefix="eval"
    )

    # Create an instance of the table evaluator
    evaluator = evaluators.AggregatedEvaluator(
        [evaluators.PascalEvaluator, evaluators.ICDAR19Evaluator]
    )

    # For each batch of (images, targets) pairs
    for images, targets in dataloader_wrapper:

        # Transfer to device
        images = list(image.to(params.generic.device) for image in images)

        # Get model outputs
        model_time = time.time()
        outputs = model(images)
        model_time = time.time() - model_time

        # Accumulate outputs in the evaluator
        evaluator.update(list(zip(outputs, targets)))

        # Update time metrics
        metric_logger.update(model_time=model_time)

    # Compute evaluation metrics
    evaluator_time = time.time()
    metrics = evaluator.evaluate()
    metric_logger.update(**metrics)
    evaluator_time = time.time() - evaluator_time

    # Update time metrics and evetually log metrics to wandb
    metric_logger.update(evaluator_time=evaluator_time)
    if params.generic.wandb.enabled:
        metric_logger.wandb_log("eval", {'evaluator_time': evaluator_time})

    # Print evaluation info
    print("Averaged stats:", metric_logger)
