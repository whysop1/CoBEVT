'''
import torch
import pytorch_lightning as pl

from . import get_dataset_module_by_name


class DataModule(pl.LightningDataModule):
    def __init__(self, dataset: str, data_config: dict, loader_config: dict):
        super().__init__()

        self.get_data = get_dataset_module_by_name(dataset).get_data

        self.data_config = data_config
        self.loader_config = loader_config

    def get_split(self, split, loader=True, shuffle=False):
        datasets = self.get_data(split=split, **self.data_config)

        if not loader:
            return datasets

        dataset = torch.utils.data.ConcatDataset(datasets)

        loader_config = dict(self.loader_config)

        if loader_config['num_workers'] == 0:
            loader_config['prefetch_factor'] = 2

        return torch.utils.data.DataLoader(dataset, shuffle=shuffle, **loader_config)

    def train_dataloader(self, shuffle=True):
        return self.get_split('train', loader=True, shuffle=shuffle)

    def val_dataloader(self, shuffle=True):
        return self.get_split('val', loader=True, shuffle=shuffle)
'''

class DataModule(pl.LightningDataModule):
    def __init__(self, dataset: str, data_config: dict, loader_config: dict):
        super().__init__()

        self.get_data = get_dataset_module_by_name(dataset).get_data

        self.data_config = data_config
        self.loader_config = loader_config

    def get_split(self, split, loader=True, shuffle=False):
        datasets = self.get_data(split=split, **self.data_config)

        if not loader:
            return datasets

        dataset = torch.utils.data.ConcatDataset(datasets)

        loader_config = dict(self.loader_config)

        # num_workers 값이 너무 크면 경고를 방지하도록 설정
        if loader_config['num_workers'] > 4:  # 4보다 클 경우, 4로 설정
            print("Warning: num_workers is too high. Reducing to 4.")
            loader_config['num_workers'] = 4

        return torch.utils.data.DataLoader(dataset, shuffle=shuffle, **loader_config)

    def train_dataloader(self, shuffle=True):
        return self.get_split('train', loader=True, shuffle=shuffle)

    def val_dataloader(self, shuffle=True):
        return self.get_split('val', loader=True, shuffle=shuffle)
