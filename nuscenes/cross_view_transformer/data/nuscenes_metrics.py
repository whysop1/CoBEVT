
import numpy as np
from pyquaternion import Quaternion

def compute_scene_metrics(nusc, sample_token, view_matrix, pose_inverse):
    """
    OpenCV 렌더링 없이 수학적 계산만으로 object_count와 variance를 산출합니다.
    """
    # 1. 샘플 및 어노테이션 데이터 가져오기
    sample_record = nusc.get('sample', sample_token)
    ann_tokens = sample_record['anns']
    
    # 2. 메트릭 초기화
    object_count = 0
    centers_bev = []
    
    # 가중치 2배 적용 대상
    dangerous_classes = ['pedestrian', 'bus', 'motorcycle', 'bicycle', 'truck']
    
    # BEV 변환을 위한 행렬 준비
    V = np.array(view_matrix)
    M_inv = np.array(pose_inverse)
    S = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 1],
    ])

    for ann_token in ann_tokens:
        ann = nusc.get('sample_annotation', ann_token)
        cat_name = ann['category_name']
        
        # 'vehicle'이나 'human'이 포함된 객체만 처리 (필요에 따라 수정 가능)
        if not any(c in cat_name for c in ['vehicle', 'human', 'bicycle', 'motorcycle']):
            continue

        # 3. 좌표 변환 (3D -> BEV 2D)
        # translation: [x, y, z]
        pos = np.array(ann['translation']).reshape(3, 1)
        pos_homo = np.vstack((pos, [1.0]))  # 4x1
        
        # BEV 좌표계로 투영
        p_bev = V @ S @ M_inv @ pos_homo  # 3x1
        center_x, center_y = p_bev[0, 0], p_bev[1, 0]
        
        # BEV 영역(예: 200x200) 안에 있는지 확인 (범위 밖 객체 제외)
        if 0 <= center_x < 200 and 0 <= center_y < 200:
            centers_bev.append([center_x, center_y])
            
            # 4. 가중치 카운팅
            is_dangerous = any(dc in cat_name for dc in dangerous_classes)
            object_count += 2 if is_dangerous else 1

    # 5. 분산 계산
    if len(centers_bev) > 1:
        centers_bev = np.array(centers_bev)
        object_distribution_var = np.var(centers_bev, axis=0).mean()
    else:
        object_distribution_var = 0.0

    return object_count, object_distribution_var
