'''
import torch
import json
import hydra
import cv2
import numpy as np

from pathlib import Path
from tqdm import tqdm

from cross_view_transformer.data.transforms import LoadDataTransform
from cross_view_transformer.common import setup_config, setup_data_module, setup_viz


def setup(cfg):
    # Don't change these
    cfg.data.dataset = cfg.data.dataset.replace('_generated', '')
    cfg.data.augment = 'none'
    cfg.loader.batch_size = 1
    cfg.loader.persistent_workers = True
    cfg.loader.drop_last = False
    cfg.loader.shuffle = False

    # Uncomment to debug errors hidden by multiprocessing
    # cfg.loader.num_workers = 0
    # cfg.loader.prefetch_factor = 2
    # cfg.loader.persistent_workers = False


@hydra.main(config_path=Path.cwd() / 'config', config_name='config.yaml')
def main(cfg):
    """
    Creates the following dataset structure

    cfg.data.labels_dir/
        01234.json
        01234/
            bev_0001.png
            bev_0002.png
            ...

    If the 'visualization' flag is passed in,
    the generated data will be loaded from disk and shown on screen
    """
    setup_config(cfg, setup)

    data = setup_data_module(cfg)
    viz_fn = None

    if 'visualization' in cfg:
        viz_fn = setup_viz(cfg)
        load_xform = LoadDataTransform(cfg.data.dataset_dir, cfg.data.labels_dir,
                                       cfg.data.image, cfg.data.num_classes)

    labels_dir = Path(cfg.data.labels_dir)
    labels_dir.mkdir(parents=False, exist_ok=True)

    for split in ['train', 'val']:
        print(f'Generating split: {split}')

        for episode in tqdm(data.get_split(split, loader=False), position=0, leave=False):
            scene_dir = labels_dir / episode.scene_name
            scene_dir.mkdir(exist_ok=True, parents=False)

            loader = torch.utils.data.DataLoader(episode, collate_fn=list, **cfg.loader)
            info = []

            for i, batch in enumerate(tqdm(loader, position=1, leave=False)):
                info.extend(batch)

                # Load data from disk to test if it was saved correctly
                if i == 0 and viz_fn is not None:
                    unbatched = [load_xform(s) for s in batch]
                    rebatched = torch.utils.data.dataloader.default_collate(unbatched)

                    viz = np.vstack(viz_fn(rebatched))

                    cv2.imshow('debug', cv2.cvtColor(viz, cv2.COLOR_RGB2BGR))
                    cv2.waitKey(1)

            # Write all info for loading to json
            scene_json = labels_dir / f'{episode.scene_name}.json'
            scene_json.write_text(json.dumps(info))



if __name__ == '__main__':
    main()

'''

import torch
import json
import hydra
import cv2
import numpy as np

from pathlib import Path
from tqdm import tqdm

from cross_view_transformer.data.transforms import LoadDataTransform
from cross_view_transformer.common import setup_config, setup_data_module, setup_viz

# 추가된 부분: NuScenesDataset 임포트
from nuscenes_dataset import NuScenesDataset


def setup(cfg):
    # Don't change these
    cfg.data.dataset = cfg.data.dataset.replace('_generated', '')
    cfg.data.augment = 'none'
    cfg.loader.batch_size = 1
    cfg.loader.persistent_workers = True
    cfg.loader.drop_last = False
    cfg.loader.shuffle = False

    # Uncomment to debug errors hidden by multiprocessing
    # cfg.loader.num_workers = 0
    # cfg.loader.prefetch_factor = 2
    # cfg.loader.persistent_workers = False


@hydra.main(config_path=Path.cwd() / 'config', config_name='config.yaml')
def main(cfg):
    """
    Creates the following dataset structure

    cfg.data.labels_dir/
        01234.json
        01234/
            bev_0001.png
            bev_0002.png
            ...

    If the 'visualization' flag is passed in,
    the generated data will be loaded from disk and shown on screen
    """
    setup_config(cfg, setup)

    data = setup_data_module(cfg)
    viz_fn = None

    if 'visualization' in cfg:
        viz_fn = setup_viz(cfg)
        load_xform = LoadDataTransform(cfg.data.dataset_dir, cfg.data.labels_dir,
                                       cfg.data.image, cfg.data.num_classes)

    labels_dir = Path(cfg.data.labels_dir)
    labels_dir.mkdir(parents=False, exist_ok=True)

    for split in ['train', 'val']:
        print(f'Generating split: {split}')

        for episode in tqdm(data.get_split(split, loader=False), position=0, leave=False):
            scene_dir = labels_dir / episode.scene_name
            scene_dir.mkdir(exist_ok=True, parents=False)

            loader = torch.utils.data.DataLoader(episode, collate_fn=list, **cfg.loader)
            info = []

            # NuScenesDataset 인스턴스 생성 (1번 파일에서 이걸 사용해야 object_count를 얻을 수 있음)
            helper = NuScenesSingleton(cfg.data.dataset_dir, 'v1.0-trainval')  # 또는 cfg에서 'version' 값이 있으면 그것을 사용
            dataset = NuScenesDataset(episode.scene_name, episode.scene_record, helper, transform=None)

            for i, batch in enumerate(tqdm(loader, position=1, leave=False)):
                info.extend(batch)

                # 각 배치마다 object_count 계산
                for sample in batch:
                    # 각 샘플에서 dynamic objects의 수를 세기 위해 get_dynamic_objects 호출
                    anns_vehicle = dataset.get_annotations_by_category(sample, ['vehicle'])[0]
                    _, _, object_count = dataset.get_dynamic_objects(sample, anns_vehicle)

                    sample['object_count'] = object_count  # 샘플에 object_count 추가

                # Load data from disk to test if it was saved correctly
                if i == 0 and viz_fn is not None:
                    unbatched = [load_xform(s) for s in batch]
                    rebatched = torch.utils.data.dataloader.default_collate(unbatched)

                    viz = np.vstack(viz_fn(rebatched))

                    cv2.imshow('debug', cv2.cvtColor(viz, cv2.COLOR_RGB2BGR))
                    cv2.waitKey(1)

            # Write all info for loading to json
            scene_json = labels_dir / f'{episode.scene_name}.json'
            scene_json.write_text(json.dumps(info))


if __name__ == '__main__':
    main()
