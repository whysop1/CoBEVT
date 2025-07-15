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

from shapely.geometry import MultiPolygon

from .common import get_split, get_pose, INTERPOLATION
from .transforms import Sample, LoadDataTransform

STATIC = ['lane', 'road_segment']
DIVIDER = ['road_divider', 'lane_divider']
DYNAMIC = [
    'car', 'truck', 'bus',
    'trailer', 'construction',
    'pedestrian',
    'motorcycle', 'bicycle',
]

CLASSES = STATIC + DIVIDER + DYNAMIC
NUM_CLASSES = len(CLASSES)


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
    def __init__(self, scene_name, labels_dir, transform=None):
        self.samples = json.loads((Path(labels_dir) / f'{scene_name}.json').read_text())
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        data = Sample(**self.samples[idx])

        if hasattr(data, 'pose_inverse') and hasattr(data, 'pose'):
            anns = getattr(data, 'annotations', None)
            if anns is not None:
                try:
                    _, _, object_count = self.get_dynamic_objects(data, anns)
                    data.object_count = object_count
                except Exception as e:
                    print(f"[Warning] Failed to count objects: {e}")
                    data.object_count = 0
            else:
                data.object_count = 0
        else:
            data.object_count = 0

        if self.transform is not None:
            data = self.transform(data)

        return data

    def get_dynamic_objects(self, sample, annotations):
        h, w = 200, 200  # BEV shape is assumed fixed
        segmentation = np.zeros((h, w), dtype=np.uint8)
        center_score = np.zeros((h, w), dtype=np.float32)
        center_offset = np.zeros((h, w, 2), dtype=np.float32)
        center_ohw = np.zeros((h, w, 4), dtype=np.float32)
        buf = np.zeros((h, w), dtype=np.uint8)

        visibility = np.full((h, w), 255, dtype=np.uint8)
        coords = np.stack(np.meshgrid(np.arange(w), np.arange(h)), -1).astype(np.float32)
        object_count = 0

        for ann, p in zip(annotations, self.convert_to_box(sample, annotations)):
            box = p[:2, :4]
            center = p[:2, 4]
            front = p[:2, 5]
            left = p[:2, 6]

            buf.fill(0)
            cv2.fillPoly(buf, [box.round().astype(np.int32).T], 1, INTERPOLATION)
            mask = buf > 0

            if not np.count_nonzero(mask):
                continue

            sigma = 1
            segmentation[mask] = 255
            center_offset[mask] = center[None] - coords[mask]
            center_score[mask] = np.exp(-(center_offset[mask] ** 2).sum(-1) / (sigma ** 2))
            center_ohw[mask, 0:2] = ((front - center) / (np.linalg.norm(front - center) + 1e-6))[None]
            center_ohw[mask, 2:3] = np.linalg.norm(front - center)
            center_ohw[mask, 3:4] = np.linalg.norm(left - center)
            visibility[mask] = ann.get('visibility_token', 255)

            object_count += 1

        segmentation = np.float32(segmentation[..., None])
        center_score = center_score[..., None]
        result = np.concatenate((segmentation, center_score, center_offset, center_ohw), 2)

        return result, visibility, object_count

    def convert_to_box(self, sample, annotations):
        from nuscenes.utils import data_classes

        V = np.array(sample.view)
        M_inv = np.array(sample.pose_inverse)
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

