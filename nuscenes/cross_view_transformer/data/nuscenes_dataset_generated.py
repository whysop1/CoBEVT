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
import numpy as np
import cv2

from pathlib import Path
from functools import lru_cache
from pyquaternion import Quaternion

from .common import get_split, get_view_matrix, get_pose  # get_pose 추가
from .transforms import Sample, LoadDataTransform

from nuscenes.utils import data_classes  # 내부에서 필요
from shapely.geometry import MultiPolygon

INTERPOLATION = cv2.INTER_NEAREST

DYNAMIC = [
    'car', 'truck', 'bus',
    'trailer', 'construction',
    'pedestrian',
    'motorcycle', 'bicycle',
]


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

        # object_count 계산
        if hasattr(data, "aux") and hasattr(data, "visibility"):
            try:
                object_count = self.compute_object_count(data)
                data.object_count = object_count
            except Exception as e:
                print(f"[Warning] object_count 계산 실패: {e}")
                data.object_count = 0  # fallback

        if self.transform is not None:
            data = self.transform(data)

        return data

    def compute_object_count(self, data):
        """aux 정보로부터 BEV 상에서 객체 수를 유추"""
        if not hasattr(data, 'aux') or data.aux is None:
            return 0

        # aux: (H, W, C) 형태 (C >= 2 이상일 것)
        aux = data.aux
        center_score = aux[..., 1]  # center score channel
        threshold = 0.5  # 감지된 객체로 판단할 최소 score

        # 연산량 줄이기 위해 threshold 초과 영역만 봄
        mask = center_score > threshold
        num_objects = np.count_nonzero(mask)

        return int(num_objects)


# 추후 다른 용도로 필요할 수 있으므로, 변환 함수도 추가해 둠
def convert_to_box(sample, annotations, view_matrix):
    V = view_matrix
    M_inv = np.array(sample['pose_inverse'])
    S = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 1],
    ])

    for a in annotations:
        box = data_classes.Box(a['translation'], a['size'], Quaternion(a['rotation']))

        corners = box.bottom_corners()
        center = corners.mean(-1)
        front = (corners[:, 0] + corners[:, 1]) / 2.0
        left = (corners[:, 0] + corners[:, 3]) / 2.0

        p = np.concatenate((corners, np.stack((center, front, left), -1)), -1)
        p = np.pad(p, ((0, 1), (0, 0)), constant_values=1.0)
        p = V @ S @ M_inv @ p

        yield p
