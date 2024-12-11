import cv2
import dlib
import numpy as np
from PIL import Image
import torch
from sklearn.preprocessing import StandardScaler
from transformers import (
    AutoImageProcessor,
    SuperPointForKeypointDetection,
)
import numpy as np

processor = AutoImageProcessor.from_pretrained("magic-leap-community/superpoint")
superpoint_model = SuperPointForKeypointDetection.from_pretrained(
    "magic-leap-community/superpoint"
)
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")  
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def extract_superpoint_features(face_img_pil, processor, superpoint_model, device, max_num_nodes=500, feature_dim=256):
    """
    使用 SuperPoint 提取特徵點和描述子。

    Args:
        face_img_pil (PIL.Image): 臉部圖像 (RGB)。
        processor: SuperPoint 的處理器。
        superpoint_model: SuperPoint 模型。
        device: 計算設備 (CPU 或 GPU)。
        max_num_nodes (int): 最大節點數量。
        feature_dim (int): 描述子維度。

    Returns:
        descriptors (np.ndarray): 特徵描述子。
        keypoint_coords (np.ndarray): 特徵點座標。
    """
    inputs = processor(face_img_pil, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = superpoint_model(**inputs)

    image_mask = outputs.mask[0]
    image_indices = torch.nonzero(image_mask).squeeze()
    if image_indices.numel() == 0:
        print("No keypoints detected.")
        return None, None

    keypoints = outputs.keypoints[0][image_indices]
    descriptors = outputs.descriptors[0][image_indices]

    keypoint_coords = keypoints.cpu().numpy()
    descriptors = descriptors.cpu().numpy()

    # 確保 descriptors 為二維
    if descriptors.ndim == 1:
        descriptors = descriptors.reshape(1, -1)

    scaler = StandardScaler()
    descriptors = scaler.fit_transform(descriptors)

    if descriptors.shape[0] > max_num_nodes:
        indices = np.random.choice(descriptors.shape[0], max_num_nodes, replace=False)
        descriptors = descriptors[indices]
        keypoint_coords = keypoint_coords[indices]

    return descriptors, keypoint_coords

def extract_superpoint_features_single_region(img_pil, processor, superpoint_model, device, max_num_nodes=500, feature_dim=256):
    inputs = processor(img_pil, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = superpoint_model(**inputs)

    image_mask = outputs.mask[0]
    image_indices = torch.nonzero(image_mask).squeeze()
    if image_indices.numel() == 0:
        return None, None

    keypoints = outputs.keypoints[0][image_indices]
    descriptors = outputs.descriptors[0][image_indices]

    keypoint_coords = keypoints.cpu().numpy()   # 可能為 (n,2) 或 (2,)
    descriptors = descriptors.cpu().numpy()     # 可能為 (n,d) 或 (d,)

    # 確保 descriptors 為二維
    if descriptors.ndim == 1:
        descriptors = descriptors.reshape(1, -1)

    # 確保 keypoint_coords 為二維 (n, 2)
    if keypoint_coords.ndim == 1:
        keypoint_coords = keypoint_coords.reshape(1, 2)

    scaler = StandardScaler()
    descriptors = scaler.fit_transform(descriptors)

    if descriptors.shape[0] > max_num_nodes:
        indices = np.random.choice(descriptors.shape[0], max_num_nodes, replace=False)
        descriptors = descriptors[indices]
        keypoint_coords = keypoint_coords[indices]

    return descriptors, keypoint_coords

def extract_face_keypoints(img_pth):
    img = cv2.imread(img_pth)
    if img is None:
        raise ValueError(f"Unable to load image at path: {img_pth}")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    if len(faces) == 0:
        print(f"No faces detected in image {img_pth}.")
        return None, None
    face = faces[0]
    landmarks = predictor(gray, face)
    points = np.array([(landmarks.part(n).x, landmarks.part(n).y) for n in range(68)], dtype=np.int32)
    return gray, points

def extract_region(gray, points, point_indices):
    region_points = points[point_indices]
    mask = np.zeros_like(gray)
    cv2.fillConvexPoly(mask, region_points, 255)
    region_img = cv2.bitwise_and(gray, gray, mask=mask)
    region_img_pil = Image.fromarray(region_img).convert("RGB")
    return region_img_pil

def extract_eight_regions_and_features(img_pth, processor, superpoint_model, device,
                                       max_num_nodes=500, feature_dim=256):
    """
    將臉部分成 8 個區域，分別提取特徵。
    """
    gray, points = extract_face_keypoints(img_pth)
    if gray is None:
        return None

    # 以下區域定義為示範，可調整:
    regions = {
        "left_eyebrow": np.arange(22, 27),
        "right_eyebrow": np.arange(17, 22),
        "left_eye": np.arange(42, 48),
        "right_eye": np.arange(36, 42),
        "nose": np.arange(27, 36),
        "upper_mouth": np.arange(48, 55),
        "lower_mouth": np.arange(55, 68),
        "face_contour": np.arange(0, 17)
    }

    features_dict = {}
    for region_name, idxs in regions.items():
        region_img_pil = extract_region(gray, points, idxs)
        des, kp = extract_superpoint_features(region_img_pil, processor, superpoint_model, device, max_num_nodes, feature_dim)
        features_dict[region_name] = {
            "descriptors": des,
            "keypoints": kp
        }

    return features_dict


def extract_face_region(img_pth, processor=processor, superpoint_model=superpoint_model, device=device, max_num_nodes=500, feature_dim=256):
    """
    將臉分成多個區域，提取各區域特徵並合併回傳。
    若無法偵測臉則回傳(None, None)
    """
    gray, points = extract_face_keypoints(img_pth)
    if gray is None:
        return None, None

    # 您可在此定義8個或更多區域，以下僅示範4個區域
    regions = {
        "left_eye": np.arange(42, 48),
        "right_eye": np.arange(36, 42),
        "nose": np.arange(27, 36),
        "face_contour": np.arange(0, 68)
    }

    all_des = []
    all_kp = []

    for r_name, r_idxs in regions.items():
        region_img_pil = extract_region(gray, points, r_idxs)
        des, kp = extract_superpoint_features_single_region(region_img_pil, processor, superpoint_model, device, max_num_nodes, feature_dim)

        # 確保 des 和 kp 不為 None，且數量匹配
        if des is not None and kp is not None and des.shape[0] == kp.shape[0]:
            # 若某區域只有一個描述子也能正常處理
            all_des.append(des)
            all_kp.append(kp)

    # 若全部區域都無特徵
    if len(all_des) == 0:
        return None, None

    # 確保 concat 前，各區 array 形狀一致，如: (n_i, d)
    # 只要都是同樣的第二維(feature_dim)就不會有問題
    combined_descriptors = np.concatenate(all_des, axis=0)
    combined_keypoints = np.concatenate(all_kp, axis=0)

    return combined_descriptors, combined_keypoints
