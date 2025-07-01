'''
import torch

from hydra.utils import instantiate
from omegaconf import OmegaConf, DictConfig
from torchmetrics import MetricCollection
from pathlib import Path

from .model.model_module import ModelModule
from .data.data_module import DataModule
from .losses import MultipleLoss

from collections.abc import Callable
from typing import Tuple, Dict, Optional


def setup_config(cfg: DictConfig, override: Optional[Callable] = None):
    OmegaConf.set_struct(cfg, False)

    if override is not None:
        override(cfg)

    OmegaConf.resolve(cfg)
    OmegaConf.set_struct(cfg, True)

    save_dir = Path(cfg.experiment.save_dir)
    save_dir.mkdir(parents=False, exist_ok=True)


def setup_network(cfg: DictConfig):
    return instantiate(cfg.model)


def setup_model_module(cfg: DictConfig) -> ModelModule:
    backbone = setup_network(cfg)
    loss_func = MultipleLoss(instantiate(cfg.loss))
    metrics = MetricCollection({k: v for k, v in instantiate(cfg.metrics).items()})

    model_module = ModelModule(backbone, loss_func, metrics,
                               cfg.optimizer, cfg.scheduler,
                               cfg=cfg)

    return model_module


def setup_data_module(cfg: DictConfig) -> DataModule:
    return DataModule(cfg.data.dataset, cfg.data, cfg.loader)


def setup_viz(cfg: DictConfig) -> Callable:
    return instantiate(cfg.visualization)


def setup_experiment(cfg: DictConfig) -> Tuple[ModelModule, DataModule, Callable]:
    model_module = setup_model_module(cfg)
    data_module = setup_data_module(cfg)
    viz_fn = setup_viz(cfg)

    return model_module, data_module, viz_fn


def load_backbone(checkpoint_path: str, prefix: str = 'backbone'):
    checkpoint = torch.load(checkpoint_path)

    cfg = DictConfig(checkpoint['hyper_parameters'])

    cfg = OmegaConf.to_object(checkpoint['hyper_parameters'])
    # cfg['model']['encoder']['backbone']['image_height'] = cfg['model']['encoder']['backbone'].pop('input_height')
    # cfg['model']['encoder']['backbone']['image_width'] = cfg['model']['encoder']['backbone'].pop('input_width')
    # cfg['model']['encoder']['cross_view'].pop('spherical')
    # cfg['model']['encoder']['bev_embedding']['sigma'] = 1.0
    # cfg['model']['encoder']['bev_embedding']['offset'] = 0.0
    cfg = DictConfig(cfg)

    state_dict = remove_prefix(checkpoint['state_dict'], prefix)

    backbone = setup_network(cfg)
    backbone.load_state_dict(state_dict)

    return backbone


def remove_prefix(state_dict: Dict, prefix: str) -> Dict:
    result = dict()

    for k, v in state_dict.items():
        tokens = k.split('.')

        if tokens[0] == prefix:
            tokens = tokens[1:]

'''


import torch

from hydra.utils import instantiate
from omegaconf import OmegaConf, DictConfig
from torchmetrics import MetricCollection
from pathlib import Path

from .model.model_module import ModelModule
from .data.data_module import DataModule
from .losses import MultipleLoss

from collections.abc import Callable
from typing import Tuple, Dict, Optional


def setup_config(cfg: DictConfig, override: Optional[Callable] = None):
    OmegaConf.set_struct(cfg, False)

    if override is not None:
        override(cfg)

    OmegaConf.resolve(cfg)
    OmegaConf.set_struct(cfg, True)

    save_dir = Path(cfg.experiment.save_dir)
    save_dir.mkdir(parents=False, exist_ok=True)


def setup_network(cfg: DictConfig):
    print("cfg:", cfg)
    # 여기서 cfg가 DictConfig이고, 'model' 키를 포함하면 안쪽까지 들어감
    if '_target_' not in cfg and 'model' in cfg:
        cfg = cfg['model']
    
    print("Instantiating model with _target_:", cfg['_target_'])
    return instantiate(cfg)



def setup_model_module(cfg: DictConfig) -> ModelModule:
    model = setup_network(cfg.model['model'])  # 'backbone' -> 'model'로 변수명 변경
    loss_func = MultipleLoss(instantiate(cfg.loss))
    metrics = MetricCollection({k: v for k, v in instantiate(cfg.metrics).items()})

    model_module = ModelModule(
        model=model,                           # 핵심 수정: model=...
        loss_func=loss_func,
        metrics=metrics,
        optimizer_args=cfg.optimizer,
        scheduler_args=cfg.scheduler,
        cfg=cfg
    )

    return model_module



def setup_data_module(cfg: DictConfig) -> DataModule:
    return DataModule(cfg.data.dataset, cfg.data, cfg.loader)


def setup_viz(cfg: DictConfig) -> Callable:
    return instantiate(cfg.visualization)


def setup_experiment(cfg: DictConfig) -> Tuple[ModelModule, DataModule, Callable]:
    model_module = setup_model_module(cfg)
    data_module = setup_data_module(cfg)
    viz_fn = setup_viz(cfg)

    return model_module, data_module, viz_fn


import torch
from omegaconf import DictConfig, OmegaConf

# load_backbone 함수 예시
def load_backbone(checkpoint_path: str, prefix: str = 'backbone'):
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    cfg = checkpoint['hyper_parameters']

    if not isinstance(cfg, DictConfig):
        cfg = OmegaConf.create(cfg)

    # 여기서 'model' 키가 중첩되어 있음
    print("cfg.model:", cfg.model)  # {'model': {...}}
    print("_target_ in cfg.model?", '_target_' in cfg.model)  # False
    print("_target_ in cfg.model['model']?", '_target_' in cfg.model['model'])  # True

    state_dict = remove_prefix(checkpoint['state_dict'], prefix)

    # 실제 모델 설정은 cfg.model['model'] 이므로
    backbone = setup_network(cfg.model['model'])
    print("Backbone type after instantiate:", type(backbone))

    backbone.load_state_dict(state_dict)

    return backbone





def remove_prefix(state_dict: Dict, prefix: str) -> Dict:
    result = dict()

    for k, v in state_dict.items():
        tokens = k.split('.')

        if tokens[0] == prefix:
            tokens = tokens[1:]


