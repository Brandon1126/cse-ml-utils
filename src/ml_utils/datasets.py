"""Holds all dataset and dataloader classes, and functions related to data preparation

"""
import torch
import torchvision
from torch.utils.data.dataset import Dataset, random_split
from torch.utils.data.dataloader import DataLoader
from torchvision import transforms
from typing import Tuple, List
from math import floor, ceil


class Custom_MNIST_Dataset(Dataset):
    def __init__(self, sliced_dataset: List[Tuple[torch.Tensor, int]]):
        self.sliced_dataset = sliced_dataset
        
    def __len__(self):
        return len(self.sliced_dataset)
    
    def __getitem__(self, index):
        return self.sliced_dataset[index]


def create_dataloaders_MNIST_fashion(batch_size: int) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Used to instantiate 2 training Dataloaders.
    Args:
        batch_size (int): batch size of each device
        allowed_classes List[int]: allowed classes
    Returns:
        Tuple[DataLoader, DataLoader, DataLoader]: tuple of training, validation, and test dataloaders
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
        # transforms.RandomAffine(20, translate=(0.20,0.20))
    ])

    train_dataset = torchvision.datasets.FashionMNIST('.', train=True, transform=transform)
    
    train_split = ceil(len(train_dataset) * 0.90)
    valid_split = floor(len(train_dataset) * 0.10)
    
    test_data = torchvision.datasets.FashionMNIST('.', train=False, transform=transform)

    generator = torch.Generator()
    generator.manual_seed(42)
    train_data, valid_data = random_split(
        train_dataset, [train_split, valid_split], generator=generator)

    train_loader = DataLoader(
        train_data,
        batch_size=batch_size,
        pin_memory=True,  # Allocates samples into page-locked memory, speeds up data transfer to GPU
        shuffle=True,  
    )

    valid_loader = DataLoader(
        valid_data,
        batch_size=batch_size,
        pin_memory=True,
    )
    
    test_loader = DataLoader(
        test_data,
        batch_size=batch_size,
        pin_memory=True,
    )
    return train_loader, valid_loader, test_loader