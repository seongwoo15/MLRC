import os
import torch
import torchvision.datasets as datasets
import random
def reduce_dataset(dataset, reduction_factor=0.1):
    reduced_size = int(len(dataset) * reduction_factor)
    indices = random.sample(range(len(dataset)), reduced_size)
    reduced_dataset = torch.utils.data.Subset(dataset, indices)
    return reduced_dataset

class MNIST:
    def __init__(self,
                 preprocess,
                 location=os.path.expanduser('~/data'),
                 batch_size=128,
                 num_workers=16):

        reduce = False
        self.train_dataset = datasets.MNIST(
            root=location,
            download=True,
            train=True,
            transform=preprocess
        )
        if reduce:
            reduced_train_dataset = reduce_dataset(self.train_dataset, reduction_factor=0.1)
            self.train_loader = torch.utils.data.DataLoader(
                reduced_train_dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=num_workers
            )
        else:
            self.train_loader = torch.utils.data.DataLoader(
                self.train_dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=num_workers
            )
        self.test_dataset = datasets.MNIST(
            root=location,
            download=True,
            train=False,
            transform=preprocess
        )

        self.test_loader = torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers
        )

        self.classnames = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']