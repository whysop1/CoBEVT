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


import numpy as np
import cv2

class DynamicObjectSegmenter:
    def __init__(self, bev_shape, interpolation=cv2.INTER_NEAREST):
        """
        bev_shape: (height, width) tuple of BEV image
        interpolation: OpenCV interpolation flag (default: nearest)
        """
        self.h, self.w = bev_shape
        self.INTERPOLATION = interpolation

        # Pre-allocate buffers once for efficiency
        self.segmentation = np.zeros((self.h, self.w), dtype=np.uint8)
        self.center_score = np.zeros((self.h, self.w), dtype=np.float32)
        self.center_offset = np.zeros((self.h, self.w, 2), dtype=np.float32)
        self.center_ohw = np.zeros((self.h, self.w, 4), dtype=np.float32)
        self.buf = np.zeros((self.h, self.w), dtype=np.uint8)
        self.visibility = np.full((self.h, self.w), 255, dtype=np.uint8)
        self.coords = np.stack(np.meshgrid(np.arange(self.w), np.arange(self.h)), -1).astype(np.float32)

    def segment(self, sample, annotations, convert_to_box_fn):
        """
        Main segmentation + object counting function.
        Arguments:
            sample: dict with sample pose etc.
            annotations: list of annotation dicts
            convert_to_box_fn: callable that converts (sample, annotations) -> generator of 3x7 arrays
        Returns:
            result (np.ndarray): segmentation and features (h, w, 1+1+2+4)
            visibility (np.ndarray): visibility map (h, w)
            object_count (int): number of objects detected
        """
        # Clear buffers
        self.segmentation.fill(0)
        self.center_score.fill(0)
        self.center_offset.fill(0)
        self.center_ohw.fill(0)
        self.buf.fill(0)
        self.visibility.fill(255)

        object_count = 0

        for ann, p in zip(annotations, convert_to_box_fn(sample, annotations)):
            box = p[:2, :4]
            center = p[:2, 4]
            front = p[:2, 5]
            left = p[:2, 6]

            self.buf.fill(0)
            cv2.fillPoly(self.buf, [box.round().astype(np.int32).T], 1, self.INTERPOLATION)
            mask = self.buf > 0

            if not np.count_nonzero(mask):
                continue

            sigma = 1
            self.segmentation[mask] = 255
            self.center_offset[mask] = center[None] - self.coords[mask]
            self.center_score[mask] = np.exp(-(self.center_offset[mask] ** 2).sum(-1) / (sigma ** 2))

            # orientation, h/2, w/2
            front_vec = front - center
            left_vec = left - center

            norm_front = np.linalg.norm(front_vec) + 1e-6
            norm_left = np.linalg.norm(left_vec) + 1e-6

            self.center_ohw[mask, 0:2] = (front_vec / norm_front)[None]
            self.center_ohw[mask, 2:3] = norm_front
            self.center_ohw[mask, 3:4] = norm_left

            self.visibility[mask] = ann['visibility_token']

            object_count += 1

        segmentation_float = np.float32(self.segmentation[..., None])
        center_score_float = self.center_score[..., None]
        result = np.concatenate((segmentation_float, center_score_float, self.center_offset, self.center_ohw), 2)

        return result, self.visibility.copy(), object_count



import json
import torch
import numpy as np
import cv2

from pathlib import Path
from .common import get_split
from .transforms import Sample, LoadDataTransform

# 아래는 DynamicObjectSegmenter 클래스를 포함했다고 가정
class DynamicObjectSegmenter:
    def __init__(self, bev_shape, interpolation=cv2.INTER_NEAREST):
        self.h, self.w = bev_shape
        self.INTERPOLATION = interpolation

        self.segmentation = np.zeros((self.h, self.w), dtype=np.uint8)
        self.center_score = np.zeros((self.h, self.w), dtype=np.float32)
        self.center_offset = np.zeros((self.h, self.w, 2), dtype=np.float32)
        self.center_ohw = np.zeros((self.h, self.w, 4), dtype=np.float32)
        self.buf = np.zeros((self.h, self.w), dtype=np.uint8)
        self.visibility = np.full((self.h, self.w), 255, dtype=np.uint8)
        self.coords = np.stack(np.meshgrid(np.arange(self.w), np.arange(self.h)), -1).astype(np.float32)

    def segment(self, sample, annotations, convert_to_box_fn):
        self.segmentation.fill(0)
        self.center_score.fill(0)
        self.center_offset.fill(0)
        self.center_ohw.fill(0)
        self.buf.fill(0)
        self.visibility.fill(255)

        object_count = 0

        for ann, p in zip(annotations, convert_to_box_fn(sample, annotations)):
            box = p[:2, :4]
            center = p[:2, 4]
            front = p[:2, 5]
            left = p[:2, 6]

            self.buf.fill(0)
            cv2.fillPoly(self.buf, [box.round().astype(np.int32).T], 1, self.INTERPOLATION)
            mask = self.buf > 0

            if not np.count_nonzero(mask):
                continue

            sigma = 1
            self.segmentation[mask] = 255
            self.center_offset[mask] = center[None] - self.coords[mask]
            self.center_score[mask] = np.exp(-(self.center_offset[mask] ** 2).sum(-1) / (sigma ** 2))

            front_vec = front - center
            left_vec = left - center
            norm_front = np.linalg.norm(front_vec) + 1e-6
            norm_left = np.linalg.norm(left_vec) + 1e-6

            self.center_ohw[mask, 0:2] = (front_vec / norm_front)[None]
            self.center_ohw[mask, 2:3] = norm_front
            self.center_ohw[mask, 3:4] = norm_left

            self.visibility[mask] = ann['visibility_token']

            object_count += 1

        segmentation_float = np.float32(self.segmentation[..., None])
        center_score_float = self.center_score[..., None]
        result = np.concatenate((segmentation_float, center_score_float, self.center_offset, self.center_ohw), 2)

        return result, self.visibility.copy(), object_count



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
    def __init__(self, scene_name, labels_dir, transform=None):
        self.samples = json.loads((Path(labels_dir) / f'{scene_name}.json').read_text())
        self.transform = transform

    def convert_to_box(self, sample, annotations):
        from nuscenes.utils import data_classes
        from pyquaternion import Quaternion

        V = np.array(sample['view'])
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

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        # ① JSON에서 바로 dict로 읽기
        d = self.samples[idx]

        # ② BEV 해상도 정보가 json에 있다고 가정
        bev_shape = (d['bev'].shape[0], d['bev'].shape[1]) if isinstance(d['bev'], np.ndarray) else (200, 200)
        segmenter = DynamicObjectSegmenter(bev_shape)

        # ③ Annotation이 json에 들어 있다고 가정
        annotations = d.get('annotations_vehicle', [])

        # ④ object_count 계산
        if annotations and 'pose' in d and 'pose_inverse' in d and 'view' in d:
            aux, visibility, object_count = segmenter.segment(d, annotations, self.convert_to_box)
            d['aux'] = aux.tolist()
            d['visibility'] = visibility.tolist()
            d['object_count'] = object_count
        else:
            d['object_count'] = 0  # fallback

        # ⑤ Sample로 변환
        data = Sample(**d)

        if self.transform is not None:
            data = self.transform(data)

        return data
