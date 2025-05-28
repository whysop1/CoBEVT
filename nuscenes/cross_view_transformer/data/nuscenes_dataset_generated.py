'''
import json
import torch

from pathlib import Path
from .common import get_split
from .transforms import Sample, LoadDataTransform


def get_data(
    dataset_dir,
    labels_dir,
    split,
    version,
    num_classes,
    augment='none',
    image=None,                         # image config
    dataset='unused',                   # ignore
    **dataset_kwargs
):
    dataset_dir = Path(dataset_dir)
    labels_dir = Path(labels_dir)

    # Override augment if not training
    augment = 'none' if split != 'train' else augment
    transform = LoadDataTransform(dataset_dir, labels_dir, image, num_classes, augment)

    # Format the split name
    split = f'mini_{split}' if version == 'v1.0-mini' else split
    split_scenes = get_split(split, 'nuscenes')

    return [NuScenesGeneratedDataset(s, labels_dir, transform=transform) for s in split_scenes]


class NuScenesGeneratedDataset(torch.utils.data.Dataset):
    """
    Lightweight dataset wrapper around contents of a JSON file

    Contains all camera info, image_paths, label_paths ...
    that are to be loaded in the transform
    """
    def __init__(self, scene_name, labels_dir, transform=None):
        self.samples = json.loads((Path(labels_dir) / f'{scene_name}.json').read_text())
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        data = Sample(**self.samples[idx])

        if self.transform is not None:
            data = self.transform(data)

        return data
'''

import json
import torch
from pathlib import Path
from .common import get_split
from .transforms import Sample, LoadDataTransform
import numpy as np

# cluster_camera_pose_id.npy 경로 (필요에 따라 수정)
CLUSTER_PATH = 'cluster_camera_pose_id.npy'

if Path(CLUSTER_PATH).exists():
    CLUSTER_ID_MAP = np.load(CLUSTER_PATH, allow_pickle=True).item()
else:
    CLUSTER_ID_MAP = {}

def get_data(
    dataset_dir,
    labels_dir,
    split,
    version,
    num_classes,
    augment='none',
    image=None,                         # image config
    dataset='unused',                   # ignore
    **dataset_kwargs
):
    dataset_dir = Path(dataset_dir)
    labels_dir = Path(labels_dir)

    # Override augment if not training
    augment = 'none' if split != 'train' else augment
    transform = LoadDataTransform(dataset_dir, labels_dir, image, num_classes, augment)

    # Format the split name
    split = f'mini_{split}' if version == 'v1.0-mini' else split
    split_scenes = get_split(split, 'nuscenes')

    return [NuScenesGeneratedDataset(s, labels_dir, transform=transform) for s in split_scenes]


class NuScenesGeneratedDataset(torch.utils.data.Dataset):
    """
    Lightweight dataset wrapper around contents of a JSON file

    Contains all camera info, image_paths, label_paths ...
    that are to be loaded in the transform
    """
    def __init__(self, scene_name, labels_dir, transform=None):
        self.samples = json.loads((Path(labels_dir) / f'{scene_name}.json').read_text())
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        data = Sample(**self.samples[idx])

        # camera_cluster_ids가 없거나 비어있으면 매핑해서 추가
        if not hasattr(data, 'camera_cluster_ids') or not data.camera_cluster_ids:
            cluster_ids = []

            # 예시로 cam_channels 라는 리스트가 카메라 토큰이라 가정
            if hasattr(data, 'cam_channels'):
                for cam_token in data.cam_channels:
                    cluster_id = CLUSTER_ID_MAP.get(cam_token, -1)
                    cluster_ids.append(cluster_id)
            else:
                # cam_channels 없으면 images 길이만큼 -1 할당
                cluster_ids = [-1] * len(getattr(data, 'images', []))

            data.camera_cluster_ids = cluster_ids

        if self.transform is not None:
            data = self.transform(data)

        return data
