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

#추가한 라이브러리
import numpy as np
import cv2


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
    def __init__(self, scene_name, labels_dir, nusc, categories, transform=None):
        self.samples = json.loads((Path(labels_dir) / f'{scene_name}.json').read_text())
        self.nusc = nusc  # NuScenes 객체를 전달받음
        self.categories = categories  # 카테고리 리스트
        self.transform = transform

    def get_category_index(self, category_name, categories):
        """카테고리 이름을 통해 카테고리 인덱스를 반환"""
        try:
            return categories.index(category_name)
        except ValueError:
            return None

    def get_annotations_by_category(self, sample, categories):
        result = [[] for _ in categories]

        for ann_token in self.nusc.get('sample', sample['token'])['anns']:
            a = self.nusc.get('sample_annotation', ann_token)
            idx = self.get_category_index(a['category_name'], categories)

            if idx is not None:
                result[idx].append(a)

        return result

    def get_dynamic_objects(self, sample, annotations):
        h, w = self.bev_shape[:2]

        segmentation = np.zeros((h, w), dtype=np.uint8)
        center_score = np.zeros((h, w), dtype=np.float32)
        center_offset = np.zeros((h, w, 2), dtype=np.float32)
        center_ohw = np.zeros((h, w, 4), dtype=np.float32)
        buf = np.zeros((h, w), dtype=np.uint8)

        visibility = np.full((h, w), 255, dtype=np.uint8)

        coords = np.stack(np.meshgrid(np.arange(w), np.arange(h)), -1).astype(np.float32)

        object_count = 0

        # annotations을 2번 코드 형식으로 가져오기
        for ann, p in zip(annotations, self.convert_to_box(sample, annotations)):
            box = p[:2, :4]
            center = p[:2, 4]
            front = p[:2, 5]
            left = p[:2, 6]

            buf.fill(0)
            cv2.fillPoly(buf, [box.round().astype(np.int32).T], 1)
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

            visibility[mask] = ann['visibility_token']

            object_count += 1

        segmentation = np.float32(segmentation[..., None])
        center_score = center_score[..., None]

        result = np.concatenate((segmentation, center_score, center_offset, center_ohw), 2)

        return result, visibility, object_count

    def convert_to_box(self, sample, annotations):
        # 1번 코드와 동일한 방식으로 Box 변환
        pass

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # 카메라 이미지를 위한 annotation 가져오기
        anns_dynamic = self.get_annotations_by_category(sample, ['DYNAMIC'])[0]

        # 객체 수 계산
        aux, visibility, object_count = self.get_dynamic_objects(sample, anns_dynamic)

        # Sample 포장
        data = Sample(
            bev=sample['bev'],  # 예시로만 추가, 실제 필드에 맞게 수정 필요
            aux=aux,
            visibility=visibility,
            object_count=object_count,
            **sample
        )

        if self.transform:
            data = self.transform(data)

        return data

