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
from nuscenes.utils import data_classes

from .common import get_split
from .transforms import Sample, LoadDataTransform

INTERPOLATION = cv2.INTER_NEAREST


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

    augment = 'none' if split != 'train' else augment
    transform = LoadDataTransform(dataset_dir, labels_dir, image, num_classes, augment)

    split = f'mini_{split}' if version == 'v1.0-mini' else split
    split_scenes = get_split(split, 'nuscenes')

    return [NuScenesGeneratedDataset(s, labels_dir, transform=transform) for s in split_scenes]


class NuScenesGeneratedDataset(torch.utils.data.Dataset):
    """
    Loads JSON and recomputes object_count exactly like NuScenesDataset
    """

    def __init__(self, scene_name, labels_dir, transform=None):
        self.samples = json.loads((Path(labels_dir) / f'{scene_name}.json').read_text())
        self.transform = transform
        self.bev_shape = (200, 200)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        # load sample as dict
        data_dict = dict(self.samples[idx])

        # Extract fields needed
        view = np.array(data_dict['view'])
        pose_inverse = np.array(data_dict['pose_inverse'])
        annotations = data_dict.get('annotations', [])

        # Compute object_count
        object_count = self.get_dynamic_objects(view, pose_inverse, annotations)

        # Add back
        data_dict['object_count'] = object_count

        # Wrap
        data = Sample(**data_dict)

        if self.transform is not None:
            data = self.transform(data)

        return data

    def get_dynamic_objects(self, view, pose_inverse, annotations):
        """
        Matches original NuScenesDataset.get_dynamic_objects() logic
        """
        h, w = self.bev_shape

        # Preallocate
        buf = np.zeros((h, w), dtype=np.uint8)
        object_count = 0

        coords = np.stack(np.meshgrid(np.arange(w), np.arange(h)), -1).astype(np.float32)

        # For each annotation, project to BEV and rasterize
        for ann, p in zip(annotations, self.convert_to_box(view, pose_inverse, annotations)):
            box = p[:2, :4]
            center = p[:2, 4]
            front = p[:2, 5]
            left = p[:2, 6]

            buf.fill(0)
            cv2.fillPoly(buf, [box.round().astype(np.int32).T], 1, INTERPOLATION)
            mask = buf > 0

            if not np.count_nonzero(mask):
                continue

            object_count += 1

        return object_count

    def convert_to_box(self, view, pose_inverse, annotations):
        """
        Matches NuScenesDataset.convert_to_box() logic
        """
        M_inv = np.array(pose_inverse)
        S = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 1],
        ])

        for ann in annotations:
            box = data_classes.Box(ann['translation'], ann['size'], Quaternion(ann['rotation']))

            corners = box.bottom_corners()         # 3 x 4
            center = corners.mean(-1)              # 3
            front = (corners[:, 0] + corners[:, 1]) / 2.0
            left = (corners[:, 0] + corners[:, 3]) / 2.0

            p = np.concatenate((corners, np.stack((center, front, left), -1)), -1)  # 3 x 7
            p = np.pad(p, ((0, 1), (0, 0)), constant_values=1.0)                    # 4 x 7
            p = view @ S @ M_inv @ p                                                # 3 x 7

            yield p
