import os
import cv2
import torch
import numpy as np
from PIL import Image

from transformers import AutoImageProcessor, SuperPointForKeypointDetection

class DescriptorExtractor:
    def __init__(self, device=None):
        # 判斷 device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # 初始化 SuperPoint (深度學習)
        # 請確保此模型已存在，若無法使用此方法請移除或更換其他深度學習模型
        self.processor = AutoImageProcessor.from_pretrained("magic-leap-community/superpoint")
        self.superpoint_model = SuperPointForKeypointDetection.from_pretrained("magic-leap-community/superpoint").to(self.device)
        self.superpoint_model.eval()

        # 初始化傳統特徵描述子 (SIFT、ORB)
        # SIFT需要OpenCV contrib包，若無請改用ORB或AKAZE等自由度高的特徵子
        if hasattr(cv2, 'SIFT_create'):
            self.sift = cv2.SIFT_create()
        else:
            self.sift = None

        self.orb = cv2.ORB_create(nfeatures=500)

    def extract_descriptors(self, img, method='superpoint'):
        """
        從輸入影像 img 中抽取特徵點與描述子。
        
        Parameters
        ----------
        img : np.ndarray (H,W) or (H,W,3)
            輸入圖像 (RGB或BGR都可以，但SuperPoint預設認為輸入為RGB PIL)
        method : str
            可選 'superpoint', 'sift', 'orb'
        
        Returns
        -------
        keypoints : np.ndarray, shape=(N,2)
            特徵點位置(x, y)
        descriptors : np.ndarray, shape=(N,D)
            對應特徵描述子
        """
        if method == 'superpoint':
            return self._extract_superpoint(img)
        elif method == 'sift':
            return self._extract_sift(img)
        elif method == 'orb':
            return self._extract_orb(img)
        else:
            raise ValueError(f"Unknown descriptor method: {method}")

    def _extract_superpoint(self, img):
        # 若輸入為BGR (OpenCV預設), 需轉為RGB
        if img.ndim == 3 and img.shape[2] == 3:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            # 若是灰階，需轉為RGB三通道
            img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

        img_pil = Image.fromarray(img_rgb)
        inputs = self.processor(img_pil, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.superpoint_model(**inputs)

        # SuperPoint輸出 keypoints: (1, N, 2), descriptors: (1, N, D)
        keypoints = outputs['keypoints'][0].cpu().numpy() # shape (N, 2)
        descriptors = outputs['descriptors'][0].cpu().numpy() # shape (N, D)

        # 若無特徵點
        if keypoints.shape[0] == 0:
            return np.empty((0,2)), np.empty((0, descriptors.shape[1])) if descriptors.ndim > 1 else np.empty((0,))

        return keypoints, descriptors

    def _extract_sift(self, img):
        # 確保有SIFT
        if self.sift is None:
            raise RuntimeError("SIFT is not available in your OpenCV installation.")

        if img.ndim == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img

        kp, des = self.sift.detectAndCompute(gray, None)
        if des is None:
            return np.empty((0,2)), np.empty((0,128))

        # kp為cv2.KeyPoint類別的list
        keypoints = np.array([kpt.pt for kpt in kp], dtype=np.float32)
        return keypoints, des

    def _extract_orb(self, img):
        if img.ndim == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img

        kp, des = self.orb.detectAndCompute(gray, None)
        if des is None:
            return np.empty((0,2)), np.empty((0,32))

        keypoints = np.array([kpt.pt for kpt in kp], dtype=np.float32)
        return keypoints, des
