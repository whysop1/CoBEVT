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
from .common import get_split, INTERPOLATION, get_pose
from .transforms import Sample, LoadDataTransform
from pyquaternion import Quaternion

# Optional: If needed, adjust dynamically
DYNAMIC = [
    'car', 'truck', 'bus',
    'trailer', 'construction',
    'pedestrian',
    'motorcycle', 'bicycle',
]

class NuScenesSingleton:
    def __init__(self, dataset_dir, version):
        self.dataroot = str(dataset_dir)
        self.nusc = self.lazy_nusc(version, self.dataroot)

    @classmethod
    def lazy_nusc(cls, version, dataroot):
        from nuscenes.nuscenes import NuScenes
        if not hasattr(cls, '_lazy_nusc'):
            cls._lazy_nusc = NuScenes(version=version, dataroot=dataroot)
        return cls._lazy_nusc

    def __new__(cls, *args, **kwargs):
        if not hasattr(cls, '_singleton'):
            obj = super().__new__(cls)
            obj.__init__(*args, **kwargs)
            cls._singleton = obj
        return cls._singleton


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

    helper = NuScenesSingleton(dataset_dir, version)

    return [
        NuScenesGeneratedDataset(s, labels_dir, helper, transform=transform)
        for s in split_scenes
    ]


class NuScenesGeneratedDataset(torch.utils.data.Dataset):
    def __init__(self, scene_name, labels_dir, helper, transform=None):
        self.samples = json.loads((Path(labels_dir) / f'{scene_name}.json').read_text())
        self.transform = transform
        self.nusc = helper.nusc

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        data_dict = self.samples[idx]
        sample = Sample(**data_dict)

        # If object_count not in JSON, calculate from NuScenes
        if not hasattr(sample, 'object_count'):
            anns_vehicle = self.get_vehicle_annotations(sample)
            object_count = self.count_objects(sample, anns_vehicle)
            sample.object_count = object_count

        if self.transform is not None:
            sample = self.transform(sample)

        return sample

    def get_vehicle_annotations(self, sample):
        anns = []
        for ann_token in self.nusc.get('sample', sample.token)['anns']:
            ann = self.nusc.get('sample_annotation', ann_token)
            if 'vehicle' in ann['category_name']:
                anns.append(ann)
        return anns

    def count_objects(self, sample, annotations):
        h, w = 200, 200
        coords = np.stack(np.meshgrid(np.arange(w), np.arange(h)), -1).astype(np.float32)
        visibility = np.full((h, w), 255, dtype=np.uint8)

        object_count = 0
        for ann, p in zip(annotations, self.convert_to_box(sample, annotations)):
            box = p[:2, :4]
            center = p[:2, 4]
            front = p[:2, 5]
            left = p[:2, 6]

            buf = np.zeros((h, w), dtype=np.uint8)
            cv2.fillPoly(buf, [box.round().astype(np.int32).T], 1, INTERPOLATION)
            mask = buf > 0

            if not np.count_nonzero(mask):
                continue

            object_count += 1
            visibility[mask] = ann['visibility_token']

        return object_count

    def convert_to_box(self, sample, annotations):
        from nuscenes.utils import data_classes

        # Prepare transforms
        M_inv = np.array(sample.pose_inverse)
        S = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 1],
        ])
        V = np.array(sample.view)

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
