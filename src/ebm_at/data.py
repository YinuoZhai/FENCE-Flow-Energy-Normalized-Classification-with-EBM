from typing import Tuple
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from ebm_at.config import Config

cfg = Config()

def get_cifar10_loaders(cfg: Config) -> Tuple[DataLoader, DataLoader]:
    """CIFAR-10 train/test dataloaders with moderate augmentations.
    """
    normalize = transforms.Normalize((0.4914, 0.4822, 0.4465),
                                     (0.2023, 0.1994, 0.2010))

    train_tf = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])
    test_tf = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])

    train_ds = datasets.CIFAR10(root=cfg.data_root, train=True, download=True, transform=train_tf)
    test_ds = datasets.CIFAR10(root=cfg.data_root, train=False, download=True, transform=test_tf)

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,
                              num_workers=cfg.num_workers, pin_memory=True, persistent_workers=True, prefetch_factor=4)
    test_loader = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False,
                             num_workers=cfg.num_workers, pin_memory=True)
    return train_loader, test_loader