from data.base import DataModuleBase
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os
import matplotlib.pyplot as plt
import numpy as np

class VehicleMakeModelDataset(DataModuleBase):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.cfg = cfg
        self.train_bs = cfg['train']['batch_size']
        self.val_bs = cfg['val']['batch_size']
        self.test_bs = cfg['test']['batch_size']

        self.train_transforms = transforms.Compose([
            transforms.Resize(cfg['model']['input_size']),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(
                degrees=cfg['dataset']['augmentation']['rotation_range']
            ),
            transforms.RandomAdjustSharpness(
                sharpness_factor=cfg['dataset']['augmentation']['sharpness_factor'], p=0.5),
            transforms.RandomGrayscale(p=0.5),
            transforms.RandomPerspective(distortion_scale=cfg['dataset']['augmentation']['distortion_scale'], p=0.5),
            transforms.RandomPosterize(bits=cfg['dataset']['augmentation']['bits'], p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=cfg['dataset']['mean'],
                                 std=cfg['dataset']['std']),
            transforms.RandomErasing(p=0.1),
        ])

        self.test_val_transforms = transforms.Compose([
            transforms.Resize(cfg['model']['input_size']),
            transforms.ToTensor(),
            transforms.Normalize(mean=cfg['dataset']['mean'],
                                 std=cfg['dataset']['std'])
        ])

        self.prepare_dataset()

    def prepare_dataset(self):
        self.train_set = datasets.ImageFolder(root=self.cfg['dataset']['train_dir'],
                                     transform=self.train_transforms)
        self.val_set = datasets.ImageFolder(root=self.cfg['dataset']['val_dir'],
                                   transform=self.test_val_transforms)
        self.test_set = datasets.ImageFolder(root=self.cfg['dataset']['test_dir'],
                                    transform=self.test_val_transforms)

    def train_dataloader(self):
        kwargs = dict(
            batch_size=self.train_bs,
            shuffle=True,
            num_workers=self.cfg['dataset']['num_workers'],
            pin_memory=True,
        )
        self.train_dl = DataLoader(self.train_set, **kwargs)
        return self.train_dl

    def val_dataloader(self):
        kwargs = dict(
            batch_size=self.val_bs,
            shuffle=False,
            num_workers=self.cfg['dataset']['num_workers'], 
            pin_memory=True
        )
        self.val_dl = DataLoader(self.val_set, **kwargs)
        return self.val_dl

    def test_dataloader(self):
        kwargs = dict(
            batch_size=self.test_bs,
            shuffle=False,
            num_workers=self.cfg['dataset']['num_workers'],
            pin_memory=True
        )
        self.test_dl = DataLoader(self.test_set, **kwargs)
        return self.test_dl

    def get_classes(self):
        return self.train_set.classes


class VehicleColorDataset(DataModuleBase):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.cfg = cfg
        self.train_bs = cfg['train']['batch_size']
        self.val_bs = cfg['val']['batch_size']
        self.test_bs = cfg['test']['batch_size']

        self.train_transforms = transforms.Compose([
            transforms.Resize(cfg['model']['input_size']),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(
                degrees=cfg['dataset']['augmentation']['rotation_range']),
            transforms.RandomAdjustSharpness(
                sharpness_factor=cfg['dataset']['augmentation']['sharpness_factor'], p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=cfg['dataset']['mean'],
                                 std=cfg['dataset']['std']),
        ])

        self.test_val_transforms = transforms.Compose([
            transforms.Resize(cfg['model']['input_size']),
            transforms.ToTensor(),
            transforms.Normalize(mean=cfg['dataset']['mean'],
                                 std=cfg['dataset']['std'])
        ])
        
        self.prepare_dataset()

    def prepare_dataset(self):
        self.train_set = datasets.ImageFolder(root=self.cfg['dataset']['train_dir'],
                                            transform=self.train_transforms)
        self.val_set = datasets.ImageFolder(root=self.cfg['dataset']['val_dir'],
                                            transform=self.test_val_transforms)
        self.test_set = datasets.ImageFolder(root=self.cfg['dataset']['test_dir'],
                                            transform=self.test_val_transforms)
    def train_dataloader(self):
        kwargs = dict(
            batch_size=self.train_bs,
            shuffle=True,
            num_workers=self.cfg['dataset']['num_workers'],
            pin_memory=True
        )
        self.train_dl = DataLoader(self.train_set, **kwargs)
        return self.train_dl

    def val_dataloader(self):
        kwargs = dict(
            batch_size=self.val_bs,
            shuffle=False,
            num_workers=self.cfg['dataset']['num_workers'], 
            pin_memory=True
        )
        self.val_dl = DataLoader(self.val_set, **kwargs)
        return self.val_dl

    def test_dataloader(self):
        kwargs = dict(
            batch_size=self.test_bs,
            shuffle=False,
            num_workers=self.cfg['dataset']['num_workers'],
            pin_memory=True
        )
        self.test_dl = DataLoader(self.test_set, **kwargs)
        return self.test_dl

    def get_classes(self):
        return self.train_set.classes