'''
from pathlib import Path
import logging
# import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import pytorch_lightning as pl
import hydra

from pytorch_lightning.strategies.ddp import DDPStrategy
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint

from cross_view_transformer.common import setup_config, setup_experiment, load_backbone
from cross_view_transformer.callbacks.gitdiff_callback import GitDiffCallback
from cross_view_transformer.callbacks.visualization_callback import VisualizationCallback


log = logging.getLogger(__name__)

CONFIG_PATH = '/content/CoBEVT/nuscenes/config'
CONFIG_NAME = 'config.yaml'


def maybe_resume_training(experiment):
    save_dir = Path(experiment.save_dir).resolve()
    checkpoints = list(save_dir.glob(f'**/{experiment.uuid}/checkpoints/*.ckpt'))

    log.info(f'Searching {save_dir}.')

    if not checkpoints:
        return None

    log.info(f'Found {checkpoints[-1]}.')

    return checkpoints[-1]


@hydra.main(config_path=CONFIG_PATH, config_name=CONFIG_NAME)
def main(cfg):
    setup_config(cfg)

    pl.seed_everything(cfg.experiment.seed, workers=True)

    Path(cfg.experiment.save_dir).mkdir(exist_ok=True, parents=False)

    # Create and load model/data
    model_module, data_module, viz_fn = setup_experiment(cfg)

    # Optionally load model
    ckpt_path = maybe_resume_training(cfg.experiment)

    if ckpt_path is not None:
        model_module.backbone = load_backbone(ckpt_path)

    # Loggers and callbacks
    logger = pl.loggers.WandbLogger(project=cfg.experiment.project,
                                    save_dir=cfg.experiment.save_dir,
                                    id=cfg.experiment.uuid)

    callbacks = [
        LearningRateMonitor(logging_interval='epoch'),
        ModelCheckpoint(filename='model',
                        every_n_train_steps=cfg.experiment.checkpoint_interval),

        VisualizationCallback(viz_fn, cfg.experiment.log_image_interval),
        GitDiffCallback(cfg)
    ]

    # Train
    trainer = pl.Trainer(logger=logger,
                         callbacks=callbacks,
                         strategy=DDPStrategy(find_unused_parameters=False),
                         **cfg.trainer)
    trainer.fit(model_module, datamodule=data_module, ckpt_path=ckpt_path)


if __name__ == '__main__':
    main()

'''

from pathlib import Path
import logging
import time
import torch
import numpy as np
import pytorch_lightning as pl
import hydra
import wandb

from pytorch_lightning.strategies.ddp import DDPStrategy
from pytorch_lightning.callbacks import LearningRateMonitor, Callback

from cross_view_transformer.common import setup_config, setup_experiment
# load_backbone은 사용하지 않으므로 제외 가능

log = logging.getLogger(__name__)

CONFIG_PATH = '/content/CoBEVT/nuscenes/config'
CONFIG_NAME = 'config.yaml'

# ==========================================
# 1. 정밀 추론 시간 측정 콜백 (GPU 동기화 포함)
# ==========================================
class InferenceTimeCallback(Callback):
    def __init__(self):
        super().__init__()
        self.epoch_times = []
        self.batch_start_time = 0.0

    def on_validation_batch_start(self, trainer, pl_module, batch, batch_idx, dataloader_idx=0):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self.batch_start_time = time.time()

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        end_time = time.time()

        elapsed = end_time - self.batch_start_time
        
        # [수정 포인트 1] 나누기(/ batch_size)를 삭제하고 elapsed(배치당 시간)를 그대로 저장
        self.epoch_times.append(elapsed)

    def on_validation_epoch_end(self, trainer, pl_module):
        if self.epoch_times:
            avg_batch_time = np.mean(self.epoch_times)
            std_batch_time = np.std(self.epoch_times)
            
            # [수정 포인트 2] 출력 문구를 "per batch"로 변경
            print(f"\n[Inference Test] Avg time per batch: {avg_batch_time:.6f}s (Std: {std_batch_time:.6f}s)")
            
            if trainer.logger:
                trainer.logger.log_metrics({
                    "test_avg_batch_inference_time": avg_batch_time,
                    "test_std_batch_inference_time": std_batch_time
                })
            self.epoch_times = []

@hydra.main(config_path=CONFIG_PATH, config_name=CONFIG_NAME)
def main(cfg):
    setup_config(cfg)
    pl.seed_everything(cfg.experiment.seed, workers=True)

    # 1. 모델 및 데이터 모듈 생성 (가중치는 이때 랜덤 초기화됨)
    model_module, data_module, viz_fn = setup_experiment(cfg)

    # [수정] 가중치 로딩 로직 전체 주석 처리 또는 삭제
    # ckpt_path = maybe_resume_training(cfg.experiment)
    # if ckpt_path is not None:
    #     model_module.backbone = load_backbone(ckpt_path)

    # 2. 로거 설정 (시간 측정값 기록용)
    logger = pl.loggers.WandbLogger(
        project=cfg.experiment.project,
        save_dir=cfg.experiment.save_dir,
        id=f"inference_test_{cfg.experiment.uuid}"
    )

    # 3. 불필요한 콜백 제거 및 측정 콜백만 추가
    # 학습이 아니므로 Checkpoint나 Visualization은 끕니다.
    callbacks = [
        InferenceTimeCallback()
    ]

    # 4. Trainer 설정
    trainer = pl.Trainer(
        logger=logger,
        callbacks=callbacks,
        # 추론 시간만 잴 때는 DDP보다 단일 GPU(auto)나 1개 전략이 측정 오차가 적습니다.
        accelerator="auto",
        devices=1, 
        **cfg.trainer
    )

    # 5. [수정] fit 대신 validate 호출 (추론만 수행)
    log.info("Starting inference time measurement with random weights...")
    trainer.validate(model=model_module, datamodule=data_module)

if __name__ == '__main__':
    main()
