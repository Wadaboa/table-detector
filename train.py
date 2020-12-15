import torch
import torchvision
import yaml

import learning_utils
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
    dataset_type = params.train_dataset
    dataset = DATASETS[dataset_type](
        roots=params.datasets.__dict__[dataset_type].path
    )
    if params.dummy.enabled:
        dataset = torch.utils.data.Subset(
            dataset, list(range(params.dummy.size))
        )
    train_size = int(params.train_split * len(dataset))
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
    utils.fix_random(params.random_seed)

    # Define datasets
    train_dataset, test_dataset = get_dataset(params)

    # Define samplers
    train_sampler = torch.utils.data.RandomSampler(train_dataset)
    test_sampler = torch.utils.data.SequentialSampler(test_dataset)
    train_batch_sampler = torch.utils.data.BatchSampler(
        train_sampler, params.batch_size, drop_last=True
    )

    # Define data loaders
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_sampler=train_batch_sampler,
        num_workers=params.workers, collate_fn=learning_utils.collate_fn
    )
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset, batch_size=1, sampler=test_sampler,
        num_workers=params.workers, collate_fn=learning_utils.collate_fn
    )

    # Get the object detector
    detector = detectors.get_detector(params, NUM_CLASSES)
    detector.to(params.device)

    # Define the optimizer
    detector_params = [p for p in detector.parameters() if p.requires_grad]
    optimizer_type = params.optimizers.type
    optimizer_params = params.optimizers.__dict__[optimizer_type].__dict__
    optimizer = OPTIMIZERS[optimizer_type](detector_params, **optimizer_params)

    # Define the learning rate scheduler
    lr_scheduler_type = params.lr_schedulers.type
    lr_scheduler_params = params.lr_schedulers.__dict__[
        lr_scheduler_type
    ].__dict__
    lr_scheduler = LR_SCHEDULERS[lr_scheduler_type](
        optimizer, **lr_scheduler_params
    )

    # Call the training/evaluation loop
    learning_utils.training_loop(
        params, detector, optimizer, lr_scheduler, train_dataloader, test_dataloader
    )


def main():
    with open('parameters.yml', 'r') as conf:
        args = yaml.load(conf, Loader=yaml.FullLoader)
    params = utils.Struct(**args)
    train(params)


if __name__ == "__main__":
    main()
