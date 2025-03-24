import logging
import pytorch_lightning as pl
import git
from pathlib import Path
from pytorch_lightning.utilities import rank_zero_only
from omegaconf import OmegaConf, DictConfig

log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent.parent
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

    @rank_zero_only
    def on_fit_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        # Git diff 가져오기
        repo = git.Repo(PROJECT_ROOT)
        diff = repo.git.diff()  # 실제 git diff를 가져옴

        # cfg를 YAML로 변환
        cfg = OmegaConf.to_yaml(self.cfg)

        # 템플릿에 diff와 cfg를 포함하여 로그 출력
        log.info(TEMPLATE.format(diff=diff, cfg=cfg))



'''
import logging
import pytorch_lightning as pl
import git

from pathlib import Path
from pytorch_lightning.utilities import rank_zero_only
from omegaconf import OmegaConf, DictConfig


log = logging.getLogger(__name__)


PROJECT_ROOT = Path(__file__).parent.parent.parent
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

    @rank_zero_only
    def on_fit_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        #diff = git.Repo(PROJECT_ROOT).git.diff()
        cfg = OmegaConf.to_yaml(self.cfg)

        log.info(TEMPLATE.format(cfg=cfg))
'''
