'''

from pathlib import Path
import subprocess
import logging
import pytorch_lightning as pl
import hydra
import sys  # ✅ 현재 파이썬 실행 파일 경로를 얻기 위해 추가

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
    # 1. cluster_camera_pose_id.npy 파일 체크 및 생성
    CLUSTER_PATH = Path('cluster_camera_pose_id.npy') 
    if not CLUSTER_PATH.exists():
        print("Cluster ID mapping not found. Generating...")

        # ✅ 현재 실행 중인 파이썬 인터프리터를 명시적으로 사용
        python_exec = sys.executable
        subprocess.run([python_exec, '/content/CoBEVT/nuscenes/tools/cluster_camera_pose.py'], check=True)
    else:
        print("Cluster ID mapping found.")

    # 2. 설정 및 시드 고정
    setup_config(cfg)
    pl.seed_everything(cfg.experiment.seed, workers=True)

    Path(cfg.experiment.save_dir).mkdir(exist_ok=True, parents=False)

    # 3. 모델 및 데이터 준비
    model_module, data_module, viz_fn = setup_experiment(cfg)

    # 4. 이전 학습 재개 체크 및 모델 로드
    ckpt_path = maybe_resume_training(cfg.experiment)
    if ckpt_path is not None:
        model_module.backbone = load_backbone(ckpt_path)

    # 5. 로그 및 콜백 정의
    logger = pl.loggers.WandbLogger(
        project=cfg.experiment.project,
        save_dir=cfg.experiment.save_dir,
        id=cfg.experiment.uuid
    )

    callbacks = [
        LearningRateMonitor(logging_interval='epoch'),
        ModelCheckpoint(
            filename='model',
            every_n_train_steps=cfg.experiment.checkpoint_interval
        ),
        VisualizationCallback(viz_fn, cfg.experiment.log_image_interval),
        GitDiffCallback(cfg)
    ]

    # 6. 트레이너 생성 및 학습 시작
    trainer = pl.Trainer(
        logger=logger,
        callbacks=callbacks,
        strategy=DDPStrategy(find_unused_parameters=False),
        **cfg.trainer
    )
    trainer.fit(model_module, datamodule=data_module, ckpt_path=ckpt_path)


if __name__ == '__main__':
    main()

'''


from pathlib import Path
import subprocess
import logging
import pytorch_lightning as pl
import hydra
import sys  # ✅ 현재 파이썬 실행 파일 경로를 얻기 위해 추가

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
    # 1. cluster_camera_pose_id.npy 파일 체크 및 생성
    CLUSTER_PATH = Path('cluster_camera_pose_id.npy') 
    if not CLUSTER_PATH.exists():
        print("Cluster ID mapping not found. Generating...")

        # ✅ 현재 실행 중인 파이썬 인터프리터를 명시적으로 사용
        python_exec = sys.executable
        subprocess.run([python_exec, '/content/CoBEVT/nuscenes/tools/cluster_camera_pose.py'], check=True)
    else:
        print("Cluster ID mapping found.")

    # 2. 설정 및 시드 고정
    setup_config(cfg)
    pl.seed_everything(cfg.experiment.seed, workers=True)

    Path(cfg.experiment.save_dir).mkdir(exist_ok=True, parents=False)

    # 3. 모델 및 데이터 준비
    model_module, data_module, viz_fn = setup_experiment(cfg)

    # 💡 디버깅: 모델 객체 타입 확인
    print(f"[DEBUG] model_module.backbone type: {type(model_module.backbone)}")


    # 4. 이전 학습 재개 체크 및 모델 로드
    ckpt_path = maybe_resume_training(cfg.experiment)
    if ckpt_path is not None:
        model_module.backbone = load_backbone(ckpt_path)

    # 5. 로그 및 콜백 정의
    logger = pl.loggers.WandbLogger(
        project=cfg.experiment.project,
        save_dir=cfg.experiment.save_dir,
        id=cfg.experiment.uuid
    )

    callbacks = [
        LearningRateMonitor(logging_interval='epoch'),
        ModelCheckpoint(
            filename='model',
            every_n_train_steps=cfg.experiment.checkpoint_interval
        ),
        VisualizationCallback(viz_fn, cfg.experiment.log_image_interval),
        GitDiffCallback(cfg)
    ]

    # 6. 트레이너 생성 및 학습 시작
    trainer = pl.Trainer(
        logger=logger,
        callbacks=callbacks,
        strategy=DDPStrategy(find_unused_parameters=False),
        **cfg.trainer
    )
    trainer.fit(model_module, datamodule=data_module, ckpt_path=ckpt_path)


if __name__ == '__main__':
    main()


