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

    scaler = StandardScaler()
    descriptors = scaler.fit_transform(descriptors)

    if descriptors.shape[0] > max_num_nodes:
        indices = np.random.choice(descriptors.shape[0], max_num_nodes, replace=False)
        descriptors = descriptors[indices]
        keypoint_coords = keypoint_coords[indices]

    return descriptors, keypoint_coords 

def extract_face_region(img_pth):
    """
    使用 dlib 檢測臉部並提取臉部區域。

    Args:
        img_pth (str): 圖像路徑。

    Returns:
        face_img_pil (PIL.Image): 提取的臉部圖像 (RGB)。
        keypoint_coords (np.ndarray): 臉部關鍵點座標 (68 個點)。
    """
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
    points = np.array(
        [(landmarks.part(n).x, landmarks.part(n).y) for n in range(68)],
        dtype=np.int32,
    )

    mask = np.zeros_like(gray)
    cv2.fillConvexPoly(mask, points, 255)
    face_img = cv2.bitwise_and(gray, gray, mask=mask)

    face_img_pil = Image.fromarray(face_img).convert("RGB")

    keypoint_coords = points  # 或者根據需要返回其他關鍵點信息

    return face_img_pil, keypoint_coords
