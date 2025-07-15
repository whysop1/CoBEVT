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

from .transforms import Sample, LoadDataTransform

import numpy as np
import cv2
from pyquaternion import Quaternion

from .common import INTERPOLATION, get_view_matrix, get_pose, get_split



class ObjectCounter:
    def __init__(self, bev_shape, view_matrix):
        self.bev_shape = bev_shape  # (h, w)
        self.view = view_matrix

    def count_objects(self, sample, annotations):
        h, w = self.bev_shape[:2]
        buf = np.zeros((h, w), dtype=np.uint8)
        coords = np.stack(np.meshgrid(np.arange(w), np.arange(h)), -1).astype(np.float32)

        object_count = 0
        for ann, p in zip(annotations, self._convert_to_box(sample, annotations)):
            box = p[:2, :4]
            center = p[:2, 4]
            front = p[:2, 5]
            left = p[:2, 6]

            buf.fill(0)
            cv2.fillPoly(buf, [box.round().astype(np.int32).T], 1, cv2.INTER_LINEAR)
            mask = buf > 0

            if not np.count_nonzero(mask):
                continue

            object_count += 1

        return object_count

    def _convert_to_box(self, sample, annotations):
        from nuscenes.utils import data_classes

        V = self.view
        M_inv = np.array(sample['pose_inverse'])
        S = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 1],
        ])

        for a in annotations:
            box = data_classes.Box(a['translation'], a['size'], Quaternion(a['rotation']))

            corners = box.bottom_corners()  # 3x4
            center = corners.mean(-1)
            front = (corners[:, 0] + corners[:, 1]) / 2.0
            left = (corners[:, 0] + corners[:, 3]) / 2.0

            p = np.concatenate((corners, np.stack((center, front, left), -1)), -1)  # 3x7
            p = np.pad(p, ((0, 1), (0, 0)), constant_values=1.0)  # 4x7
            p = V @ S @ M_inv @ p  # 3x7

            yield p


def get_data(
    dataset_dir,
    labels_dir,
    split,
    version,
    num_classes,
    augment='none',
    image=None,
    dataset='unused',
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

    # ✅ 1. BEV 설정
    bev_shape = (200, 200)
    bev_meters = {'h_meters': 100, 'w_meters': 100, 'offset': 0.0}
    view_matrix = get_view_matrix(h=bev_shape[0], w=bev_shape[1], **bev_meters)

    # ✅ 2. ObjectCounter 생성
    object_counter = ObjectCounter(bev_shape, view_matrix)

    # ✅ 3. NuScenesGeneratedDataset에 전달
    return [
        NuScenesGeneratedDataset(
            scene_name=s,
            labels_dir=labels_dir,
            transform=transform,
            object_counter=object_counter,
        ) for s in split_scenes
    ]



class NuScenesGeneratedDataset(torch.utils.data.Dataset):
    """
    Lightweight dataset wrapper around contents of a JSON file

    Contains all camera info, image_paths, label_paths ...
    that are to be loaded in the transform
    """
    def __init__(self, scene_name, labels_dir, transform=None):
        self.samples = json.loads((Path(labels_dir) / f'{scene_name}.json').read_text())
        self.transform = transform
        self.object_counter = object_counter

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        data = Sample(**self.samples[idx])

        if self.object_counter is not None:
            annotations = data['annotations'] if 'annotations' in data else []
            object_count = self.object_counter.count_objects(data, annotations)
            data['object_count'] = object_count
        
        if self.transform is not None:
            data = self.transform(data)

        return data
