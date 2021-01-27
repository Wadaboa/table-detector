import torch
import wandb
import yaml

import learning
import transforms
import detectors
import utils
from datasets import DATASETS


NUM_CLASSES = 2
OPTIMIZERS = {
    'adam': torch.optim.Adam,
    'rmsprop': torch.optim.RMSprop,
    'sgd': torch.optim.SGD
}
LR_SCHEDULERS = {
    'step': torch.optim.lr_scheduler.StepLR,
    'multi_step': torch.optim.lr_scheduler.MultiStepLR
}


def get_dataset(params):
    '''
    Return an instance of the dataset specified in the given parameters,
    splitted into training and test sets
    '''
    dataset_type = params.dataset.train
    dataset = DATASETS[dataset_type](
        roots=params.dataset.__dict__[dataset_type].path,
        device=params.generic.device
    )
    dataset = filter_dataset(dataset)
    if params.dataset.dummy.enabled:
        dataset = torch.utils.data.Subset(
            dataset, list(range(params.dataset.dummy.size))
        )
    train_size = int(params.training.train_split * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, test_size]
    )
    return train_dataset, test_dataset


def filter_dataset(dataset):
    '''
    Keep only examples of the given dataset that have annotations
    '''
    keep = []
    for i in range(len(dataset)):
        if len(dataset.find_tables(i)) > 0:
            keep.append(i)
    return torch.utils.data.Subset(dataset, keep)


def get_train_dataloader(params, train_dataset):
    '''
    Given a train dataset, return the corresponding dataloader
    '''
    train_sampler = torch.utils.data.RandomSampler(train_dataset)
    train_batch_sampler = torch.utils.data.BatchSampler(
        train_sampler, params.training.batch_size, drop_last=False
    )
    return torch.utils.data.DataLoader(
        train_dataset, batch_sampler=train_batch_sampler,
        num_workers=params.generic.workers,
        collate_fn=learning.collate_fn
    )


def get_test_dataloader(params, test_dataset):
    '''
    Given a test dataset, return the corresponding dataloader
    '''
    test_sampler = torch.utils.data.SequentialSampler(test_dataset)
    return torch.utils.data.DataLoader(
        test_dataset, batch_size=1, sampler=test_sampler,
        num_workers=params.generic.workers,
        collate_fn=learning.collate_fn
    )


def wandb_init(params, args):
    '''
    Start a wandb run (if enabled)
    '''
    if params.generic.wandb.enabled:
        wandb.init(
            project=params.generic.wandb.project,
            entity=params.generic.wandb.entity,
            config=args
        )


def wandb_watch(params, model):
    '''
    Watch the given model with wandb (if enabled)
    '''
    if params.generic.wandb.enabled:
        wandb.watch(
            model, log=params.generic.wandb.watch,
            log_freq=params.training.log_interval
        )


def wandb_finish(params):
    '''
    End the current wandb run (if enabled)
    '''
    if params.generic.wandb.enabled:
        wandb.finish()


def get_optimizer(params, model):
    '''
    Return the optimizer defined in the parameters
    '''
    optimizer_type = params.optimizers.type
    return OPTIMIZERS[optimizer_type](
        [p for p in model.parameters() if p.requires_grad],
        **params.optimizers.__dict__[optimizer_type].__dict__
    )


def get_lr_scheduler(params, optimizer):
    '''
    Return the learning rate scheduler defined in the parameters
    '''
    lr_scheduler_type = params.lr_schedulers.type
    if lr_scheduler_type in LR_SCHEDULERS:
        return LR_SCHEDULERS[lr_scheduler_type](
            optimizer,
            **params.lr_schedulers.__dict__[lr_scheduler_type].__dict__
        )
    return None


def train(params):
    '''
    Train a model with the specified parameters
    '''
    # Fix the random seed
    utils.fix_random(params.generic.random_seed)

    # Define datasets
    train_dataset, test_dataset = get_dataset(params)

    # Define data loaders
    train_dataloader = get_train_dataloader(params, train_dataset)
    test_dataloader = get_test_dataloader(params, test_dataset)

    # Get the object detector
    detector = detectors.get_detector(params, NUM_CLASSES)

    # Watch the detector model with wandb (if enabled)
    wandb_watch(params, detector)

    # Define the optimizer
    optimizer = get_optimizer(params, detector)

    # Define the learning rate scheduler
    lr_scheduler = get_lr_scheduler(params, optimizer)

    # Call the training/evaluation loop
    learning.training_loop(
        params, detector, optimizer, train_dataloader,
        test_dataloader, lr_scheduler=lr_scheduler
    )


def main():
    with open('parameters.yml', 'r') as conf:
        args = yaml.load(conf, Loader=yaml.FullLoader)
    params = utils.Struct(**args)
    wandb_init(params, args)
    train(params)
    wandb_finish(params)


if __name__ == "__main__":
    main()
