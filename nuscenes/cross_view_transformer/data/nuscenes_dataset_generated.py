
import json
import torch
import numpy as np

from pathlib import Path
from .common import get_split
from .transforms import Sample, LoadDataTransform

# cluster_camera_pose_id.npy 경로 (필요 시 절대 경로로 변경)
CLUSTER_PATH = 'cluster_camera_pose_id.npy'

# 클러스터 ID 매핑 로드
if Path(CLUSTER_PATH).exists():
    CLUSTER_ID_MAP = np.load(CLUSTER_PATH, allow_pickle=True).item()
else:
    CLUSTER_ID_MAP = {}  # 없으면 기본값 사용


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

    # training 아닐 경우 augment 끔
    augment = 'none' if split != 'train' else augment
    transform = LoadDataTransform(dataset_dir, labels_dir, image, num_classes, augment)

    # mini 버전 처리
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

        # camera_cluster_ids 필드가 없거나 비어 있으면 매핑해서 추가
        try:
            camera_cluster_ids = data.camera_cluster_ids
        except KeyError:
            camera_cluster_ids = None

        if not camera_cluster_ids:
            cluster_ids = []
            if 'cam_channels' in data:
                for cam_token in data.cam_channels:
                    cluster_id = CLUSTER_ID_MAP.get(cam_token, -1)
                    cluster_ids.append(cluster_id)
            else:
                # fallback: 이미지 수만큼 -1 채움
                cluster_ids = [-1] * len(getattr(data, 'images', []))

            data.camera_cluster_ids = cluster_ids

        if self.transform is not None:
            data = self.transform(data)

        return data

