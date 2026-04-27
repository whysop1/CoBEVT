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
from typing import List

# 기존 유틸리티 및 변환 함수 임포트
from .common import get_split
from .transforms import Sample, LoadDataTransform

# 1번 코드의 기반 클래스들 임포트
from .nuscenes_dataset import NuScenesDataset, NuScenesSingleton

# 🚀 새로 만든 최적화 계산 함수 임포트
# (파일명이 nuscenes_metrics.py이고 같은 폴더에 있을 경우)
try:
    from .nuscenes_metrics import compute_scene_metrics
except ImportError:
    from nuscenes_metrics import compute_scene_metrics

class NuScenesGeneratedDataset(torch.utils.data.Dataset):
    """
    OpenCV 렌더링 없이 수학적 계산으로 object_count를 산출하는 
    최적화된 가속 데이터셋 래퍼입니다.
    """
    def __init__(self, scene_name, labels_dir, transform=None, dataset_dir=None, version=None):
        self.samples = json.loads((Path(labels_dir) / f'{scene_name}.json').read_text())
        self.transform = transform

        # NuScenes 데이터 접근을 위한 헬퍼 생성
        self.nusc_helper = NuScenesSingleton(dataset_dir, version)

        # 해당 scene_record 찾기
        scene_record = next((s for s in self.nusc_helper.nusc.scene if s['name'] == scene_name), None)
        
        if scene_record is None:
            raise ValueError(f"Scene with name '{scene_name}' not found.")

        # 💡 중요: view_matrix(BEV 투영 행렬)를 가져오기 위해 
        # NuScenesDataset 객체를 최소한으로 생성 유지합니다.
        self.nusc_dataset = NuScenesDataset(
            scene_name=scene_name,
            scene_record=scene_record,
            helper=self.nusc_helper,
            cameras=[[0, 1, 2, 3, 4, 5]],
            bev={'h': 200, 'w': 200, 'h_meters': 100, 'w_meters': 100, 'offset': 0.0}
        )

        # 최적화된 방식으로 object_count/var 계산 후 self.samples에 저장
        self._precompute_object_counts()

    def _precompute_object_counts(self):
        """
        🚀 최적화 버전: 모든 샘플에 대해 수학적 계산으로 메트릭을 미리 산출합니다.
        """
        nusc = self.nusc_helper.nusc
        view_matrix = self.nusc_dataset.view  # BEV 변환 행렬
        
        # print(f"[{self.nusc_dataset.scene_name}] Scene Complexity 계산 중...")

        for sample_dict in self.samples:
            token = sample_dict['token']
            pose_inverse = sample_dict['pose_inverse']
            
            # 🚀 외부 함수 호출: 렌더링 과정이 없어 매우 빠름
            count, var = compute_scene_metrics(
                nusc=nusc,
                sample_token=token,
                view_matrix=view_matrix,
                pose_inverse=pose_inverse
            )
            
            sample_dict['object_count'] = count
            sample_dict['object_distribution_var'] = var

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        # 이미 object_count가 포함된 sample_dict를 가져옴
        sample_dict = self.samples[idx]
        
        # Sample 객체로 변환 (**kwargs를 통해 모든 데이터 전달)
        data = Sample(**sample_dict)
        
        if self.transform is not None:
            data = self.transform(data)

        return data

# 외부에서 데이터셋 리스트를 생성할 때 사용하는 함수
def get_data(
    dataset_dir,
    labels_dir,
    split,
    version,
    num_classes,
    augment='none',
    image=None,             # image config
    dataset='unused',       # ignore
    **dataset_kwargs
):
    dataset_dir = Path(dataset_dir)
    labels_dir = Path(labels_dir)

    # 학습이 아닐 경우 증강 비활성화
    augment = 'none' if split != 'train' else augment
    transform = LoadDataTransform(dataset_dir, labels_dir, image, num_classes, augment)

    # Split 이름 설정 (mini 버전 대응)
    split_name = f'mini_{split}' if version == 'v1.0-mini' else split
    split_scenes = get_split(split_name, 'nuscenes')

    # 해당되는 모든 scene에 대해 데이터셋 객체 생성
    return [
        NuScenesGeneratedDataset(
            s,
            labels_dir,
            transform=transform,
            dataset_dir=dataset_dir,
            version=version
        )
        for s in split_scenes if (Path(labels_dir) / f'{s}.json').exists()
    ]

