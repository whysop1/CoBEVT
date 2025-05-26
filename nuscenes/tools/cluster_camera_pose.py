# tools/cluster_camera_pose.py

import numpy as np
import os
import json
from sklearn.cluster import KMeans
from tqdm import tqdm

# NuScenes devkit import (경로에 맞게 조정)
from nuscenes.nuscenes import NuScenes

def get_camera_pose(sensor_data, ego_pose):
    translation = np.array(sensor_data['translation'])  # 3차원 위치
    rotation = np.array(sensor_data['rotation'])  # 쿼터니언
    # ego_pose와 센서 포즈 결합 (단순 위치와 방향 정보 활용)
    return np.concatenate([translation, rotation])

def main(nusc_path, version='v1.0-trainval', num_clusters=10, out_path='cluster_camera_pose_id.npy'):
    
    nusc = NuScenes(version='v1.0-trainval', dataroot=nusc_path)

    camera_poses = []
    tokens = []

    for calibrated_sensor in tqdm(nusc.calibrated_sensor, desc="Collecting sensor poses"):
        cam_pose = get_camera_pose(calibrated_sensor, ego_pose=None)
        camera_poses.append(cam_pose)
        tokens.append(calibrated_sensor['token'])

    camera_poses = np.array(camera_poses)
    kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(camera_poses)
    labels = kmeans.labels_

    token_to_cluster = {token: int(label) for token, label in zip(tokens, labels)}
    np.save(out_path, token_to_cluster)

    print(f"Saved cluster ID mapping to {out_path}")

if __name__ == "__main__":
    main(
        nusc_path='/content/drive/MyDrive/datasets/nuscenes',  # 반드시 실제 경로로 수정
        version='v1.0-trainval',
        num_clusters=10,
        out_path='cluster_camera_pose_id.npy'
    )
