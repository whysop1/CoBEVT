import logging
import pytorch_lightning as pl
import git
from pathlib import Path
from pytorch_lightning.utilities import rank_zero_only
from omegaconf import OmegaConf, DictConfig
import os

log = logging.getLogger(__name__)

# 클론된 Git 저장소 경로 설정 (구글 코랩에서 클론한 경로)
LOCAL_PATH = "/content/CoBEVT"  # 구글 코랩에서 클론한 경로

TEMPLATE = """
==================================================
{diff}
==================================================
{cfg}
==================================================
"""

class GitDiffCallback(pl.Callback):
    """
    Prints git diff and config
    """
    def __init__(self, cfg: DictConfig):
        super().__init__()

        self.cfg = cfg

        # Git 저장소 로드 (이미 클론된 경로를 사용)
        self.repo = git.Repo(LOCAL_PATH)

    @rank_zero_only
    def on_fit_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        # Git diff 가져오기
        diff = self.repo.git.diff()

        # 설정 파일을 YAML 형식으로 변환
        cfg = OmegaConf.to_yaml(self.cfg)

        # 로그 출력
        log.info(TEMPLATE.format(diff=diff, cfg=cfg))
