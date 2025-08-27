

import torch
import pytorch_lightning as pl

from . import get_dataset_module_by_name

class DataModule(pl.LightningDataModule):
    def __init__(self, dataset: str, data_config: dict, loader_config: dict):
        super().__init__()

        self.get_data = get_dataset_module_by_name(dataset).get_data
        self.data_config = data_config
        self.loader_config = loader_config

        # 🔧 num_workers를 2로 고정
        self.loader_config["num_workers"] = 2

    def get_split(self, split, loader=True, shuffle=False):
        datasets = self.get_data(split=split, **self.data_config)

        if not loader:
            return datasets

        dataset = torch.utils.data.ConcatDataset(datasets)
        loader_config = dict(self.loader_config)

        # 🔧 num_workers가 0이면 prefetch_factor 제거
        if loader_config["num_workers"] == 0:
            loader_config.pop("prefetch_factor", None)

        return torch.utils.data.DataLoader(dataset, shuffle=shuffle, **loader_config)

    def train_dataloader(self, shuffle=True):
        return self.get_split("train", loader=True, shuffle=shuffle)

    def val_dataloader(self, shuffle=True):
        return self.get_split("val", loader=True, shuffle=shuffle)
