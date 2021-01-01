'''
This module contains functions and variables to be used when
training and evaluating a PyTorch model
'''


import math
import sys
import time
import datetime
from collections import defaultdict, deque

import torch

import utils


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

    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, attr))

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(
                "{}: {}".format(name, str(meter))
            )
        return self.delimiter.join(loss_str)

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None):
        i = 0
        if not header:
            header = ''
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
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if torch.cuda.is_available():
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time),
                        memory=torch.cuda.max_memory_allocated() / MB))
                else:
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time)))
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('{} Total time: {} ({:.4f} s / it)'.format(
            header, total_time_str, total_time / len(iterable)))


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
    model.train()
    metric_logger = MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', SmoothedValue(
        window_size=1, fmt='{value:.6f}')
    )
    header = 'Epoch: [{}]'.format(epoch)

    for images, targets in metric_logger.log_every(dataloader, params.log_interval, header):
        images = torch.stack(images).to(params.device)
        targets = [
            {k: v.to(params.device) for k, v in t.items()} for t in targets
        ]

        loss_dict = model(images)
        print(loss_dict)
        losses = sum(loss for loss in loss_dict.values())
        loss_value = losses.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict)
            sys.exit(1)

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        metric_logger.update(loss=losses, **loss_dict)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    return metric_logger


def training_loop(params, model, optimizer, train_dataloader,
                  val_dataloader, lr_scheduler=None):
    start_time = time.time()
    for epoch in range(1, params.epochs + 1):
        train_one_epoch(params, model, optimizer, train_dataloader, epoch)
        if lr_scheduler is not None:
            lr_scheduler.step()
        '''
        if args.output_dir:
            utils.save_on_master({
                'model': model_without_ddp.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'args': args,
                'epoch': epoch},
                os.path.join(args.output_dir, 'model_{}.pth'.format(epoch)))
        '''

        # evaluate after every epoch
        # evaluate(model, val_dataloader)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print(f'Training time {total_time_str}')


'''
@torch.no_grad()
def evaluate(model, dataloader, device, log_interval):
    model.eval()
    metric_logger = MetricLogger(delimiter="  ")
    header = 'Test:'

    coco = get_coco_api_from_dataset(dataloader.dataset)
    iou_types = _get_iou_types(model)
    coco_evaluator = CocoEvaluator(coco, iou_types)

    for images, targets in metric_logger.log_every(dataloader, log_interval, header):
        images = list(img.to(device) for img in images)

        torch.cuda.synchronize()
        model_time = time.time()
        outputs = model(images)

        outputs = [{k: v.to(device) for k, v in t.items()}
                   for t in outputs]
        model_time = time.time() - model_time

        res = {target["image_id"].item(): output for target,
               output in zip(targets, outputs)}
        evaluator_time = time.time()
        coco_evaluator.update(res)
        evaluator_time = time.time() - evaluator_time
        metric_logger.update(model_time=model_time,
                             evaluator_time=evaluator_time)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    coco_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    coco_evaluator.accumulate()
    coco_evaluator.summarize()
    torch.set_num_threads(n_threads)
    return coco_evaluator
'''

"""
def training_loop(model, optimizer, lr_scheduler, train_dataloader, val_dataloader, device, epochs, log_interval):
    '''
    Executes the training loop for the specified number of epochs
    '''
    loop_start = time()
    losses = {
        'bbox': [],
        'class': []
    }

    for epoch in range(1, epochs + 1):
        time_start = time()
        losses = train_one_epoch(
            model, optimizer, train_dataloader, device, epoch, log_interval
        )
        '''
        loss_bb = losses_epoch_train['bbox_regression']
        losses_val_bb.append(loss_bb)

        loss_class = losses_epoch_train['classification']
        losses_val_class.append(loss_class)

        loss_sum = losses_epoch_train['sum']
        losses_val_sum.append(loss_sum)
        '''
        time_end = timer()

        lr = optimizer.param_groups[0]['lr']

        print(f'Epoch: {epoch} '
              f' Lr: {lr:.8f} '
              f' Losses Train: Sum = [{loss_sum:.4f}] Class = [{loss_class:.4f}] Boxes = [{loss_bb:.4f}]'
              f' Time one epoch (s): {(time_end - time_start):.4f} ')

        # Plot to tensorboard
        writer.add_scalar('Hyperparameters/Learning Rate', lr, epoch)
        writer.add_scalar('Metrics/Losses/Sum', loss_sum, epoch)
        writer.add_scalar('Metrics/Losses/Boxes', loss_bb, epoch)
        writer.add_scalar('Metrics/Losses/Classification', loss_class, epoch)
        writer.flush()

        if lr_scheduler:
            lr_scheduler.step()

    loop_end = time()
    time_loop = loop_end - loop_start
    if verbose:
        print(f'Time for {epochs} epochs (s): {(time_loop):.3f}')

    return {'bbox_regression': losses_val_bb,
            'classification': losses_val_class,
            'sum': losses_val_sum,
            'time': time_loop}
"""
