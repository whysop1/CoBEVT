import hydra
import numpy as np
import cv2
import matplotlib.pyplot as plt  # Matplotlib을 import

from pathlib import Path
from tqdm import tqdm

from cross_view_transformer.common import setup_config, setup_data_module, setup_viz


def setup(cfg):
    print('See training set by adding +split=train')
    print('Shuffle samples by adding +shuffle=false')

    cfg.loader.batch_size = 1

    if 'split' not in cfg:
        cfg.split = 'val'

    if 'shuffle' not in cfg:
        cfg.shuffle = False


@hydra.main(config_path='/content/CoBEVT/nuscenes/config', config_name='config.yaml')

def main(cfg):
    setup_config(cfg, setup)

    data = setup_data_module(cfg)
    viz = setup_viz(cfg)
    loader = data.get_split(cfg.split, shuffle=cfg.shuffle)

    print(f'{cfg.split}: {len(loader)} total samples')

    for batch in tqdm(loader):
        img = np.vstack(viz(batch))

        # 기존 OpenCV로 이미지를 띄우는 코드 -> Matplotlib으로 수정
        plt.figure(figsize=(10, 10))  # 이미지 크기 설정
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))  # 이미지를 Matplotlib에서 출력
        plt.axis('off')  # 축 숨기기
        plt.show()  # 이미지를 표시
        plt.pause(0.1)  # 잠깐 멈추고 다음 이미지로 넘어감


if __name__ == '__main__':
    main()



