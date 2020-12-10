import math
import sys
import time
import datetime

import torch

import utils


def train_one_epoch(params, model, optimizer, dataloader, epoch):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(
        window_size=1, fmt='{value:.6f}')
    )
    header = 'Epoch: [{}]'.format(epoch)

    for images, targets in metric_logger.log_every(dataloader, params.log_interval, header):
        images = list(image.to(params.device) for image in images)
        targets = [{k: v.to(params.device) for k, v in t.items()}
                   for t in targets]

        loss_dict = model(images, targets)
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


def training_loop(params, model, optimizer, lr_scheduler, train_dataloader, val_dataloader):
    start_time = time.time()
    for epoch in range(1, params.epochs + 1):
        train_one_epoch(params, model, optimizer, train_dataloader, epoch)
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
        #evaluate(model, val_dataloader)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print(f'Training time {total_time_str}')


'''
@torch.no_grad()
def evaluate(model, dataloader, device, log_interval):
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
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
