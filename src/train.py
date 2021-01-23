import torch
import torchvision
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
        # transforms=torchvision.transforms.Compose([
        #    transforms.Resize(
        #        (params.input_size.height, params.input_size.width)
        #    )
        # ])
    )
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


def train(params):
    '''
    Train a model with the specified parameters
    '''
    # Fix the random seed
    utils.fix_random(params.generic.random_seed)

    # Define datasets
    train_dataset, test_dataset = get_dataset(params)

    # Define samplers
    train_sampler = torch.utils.data.RandomSampler(train_dataset)
    test_sampler = torch.utils.data.SequentialSampler(test_dataset)
    train_batch_sampler = torch.utils.data.BatchSampler(
        train_sampler, params.training.batch_size, drop_last=False
    )

    # Define data loaders
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_sampler=train_batch_sampler,
        num_workers=params.generic.workers,
        collate_fn=learning.collate_fn
    )
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset, batch_size=1, sampler=test_sampler,
        num_workers=params.generic.workers,
        collate_fn=learning.collate_fn
    )

    # Get the object detector
    detector = detectors.get_detector(params, NUM_CLASSES)
    detector.to(params.generic.device)

    # Define the optimizer
    optimizer_type = params.optimizers.type
    optimizer = OPTIMIZERS[optimizer_type](
        [p for p in detector.parameters() if p.requires_grad],
        **params.optimizers.__dict__[optimizer_type].__dict__
    )

    # Define the learning rate scheduler
    lr_scheduler = None
    lr_scheduler_type = params.lr_schedulers.type
    if lr_scheduler_type in LR_SCHEDULERS:
        lr_scheduler = LR_SCHEDULERS[lr_scheduler_type](
            optimizer,
            **params.lr_schedulers.__dict__[lr_scheduler_type].__dict__
        )

    # Call the training/evaluation loop
    learning.training_loop(
        params, detector, optimizer, train_dataloader,
        test_dataloader, lr_scheduler=lr_scheduler
    )


def main():
    with open('parameters.yml', 'r') as conf:
        args = yaml.load(conf, Loader=yaml.FullLoader)
    params = utils.Struct(**args)
    train(params)


if __name__ == "__main__":
    main()
