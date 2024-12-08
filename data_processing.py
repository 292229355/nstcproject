import os
import dlib
import numpy as np
import tensorflow as tf
import cv2
from sklearn.preprocessing import StandardScaler
from similarities import scm_similarity_matrix, cosine_similarity_matrix
from adjacency import andm_adjacency, knn_adjacency, threshold_adjacency
from extract_descriptor import DescriptorExtractor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")


extractor = DescriptorExtractor()


def extract_descriptors_and_build_graph(
    img_pth, 
    max_num_nodes=500, 
    feature_dim=256, 
    eta=0.5, 
    dk=64, 
    method="ANDM_SCM", 
    k=5, 
    threshold=0.5,
    descriptor_method="superpoint"
):
    """
    Parameters
    ----------
    method : str
        Graph構建的方法，如"ANDM_SCM", "KNN_COSINE", "THRESHOLD_COSINE"等。
    descriptor_method : str
        特徵描述子方法，如"superpoint", "sift", "orb", "r2d2", "d2net"

    用命名約定:
    - ANDM_SCM: 使用SCM相似度 + ANDM
    - ANDM_COSINE: 使用Cosine相似度 + ANDM
    - KNN_COSINE: 使用Cosine相似度 + kNN鄰接
    - THRESHOLD_COSINE: 使用Cosine相似度 + thresholding
    """
    img = cv2.imread(img_pth)
    if img is None:
        return None, None, None
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = detector(gray)
    if len(faces) == 0:
        return None, None, None

    # 取第一張臉
    face = faces[0]
    landmarks = predictor(gray, face)
    points = np.array([(landmarks.part(n).x, landmarks.part(n).y) for n in range(68)], dtype=np.int32)

    mask = np.zeros_like(gray)
    cv2.fillConvexPoly(mask, points, 255)
    face_img = cv2.bitwise_and(gray, gray, mask=mask)

    # 使用指定的 descriptor_method 取得特徵描述子
    # 注意：extract_descriptors 預期輸入為RGB或BGR影像，若必要可轉RGB
    if face_img.ndim == 2:
        face_img_color = cv2.cvtColor(face_img, cv2.COLOR_GRAY2BGR)
    else:
        face_img_color = face_img

    keypoints_all, descriptors_all = extractor.extract_descriptors(face_img_color, method=descriptor_method)

    # 若沒有偵測到特徵點
    if descriptors_all.shape[0] == 0:
        return None, None, None

    # 標準化描述子
    scaler = StandardScaler()
    descriptors = scaler.fit_transform(descriptors_all)

    # 節點限制
    if descriptors.shape[0] > max_num_nodes:
        indices = np.random.choice(descriptors.shape[0], max_num_nodes, replace=False)
        descriptors = descriptors[indices]
        keypoints_all = keypoints_all[indices]

    if descriptors.shape[0] == 0:
        return None, None, None

    # 選擇相似度函數
    if "SCM" in method:
        sim_func = scm_similarity_matrix
    else:
        # 預設用cosine
        sim_func = cosine_similarity_matrix

    if method.startswith("ANDM"):
        # 使用 ANDM 構建圖
        adjacency_matrix = andm_adjacency(descriptors, sim_func, eta=eta, dk=dk)
    elif method.startswith("KNN"):
        sim_matrix = sim_func(descriptors)
        adjacency_matrix = knn_adjacency(sim_matrix, k=k)
    elif method.startswith("THRESHOLD"):
        sim_matrix = sim_func(descriptors)
        adjacency_matrix = threshold_adjacency(sim_matrix, threshold=threshold)
    else:
        # 預設用 ANDM + SCM
        adjacency_matrix = andm_adjacency(descriptors, sim_func, eta=eta, dk=dk)

    return descriptors, adjacency_matrix, keypoints_all


class DescriptorGraphDataset(tf.keras.utils.Sequence):
    def __init__(self, path, mode="train", max_num_nodes=500, feature_dim=256, eta=0.5, dk=64, batch_size=32, method="ANDM_SCM"):
        self.path = path
        self.mode = mode
        self.max_num_nodes = max_num_nodes
        self.feature_dim = feature_dim
        self.eta = eta
        self.dk = dk
        self.batch_size = batch_size
        self.method = method

        self.files = sorted([os.path.join(path, x) for x in os.listdir(path) if x.lower().endswith(".jpg") or x.lower().endswith(".png")])
        if len(self.files) == 0:
            raise FileNotFoundError(f"No image files found in {path}.")

    def __len__(self):
        return len(self.files) // self.batch_size + int((len(self.files) % self.batch_size) != 0)

    def __getitem__(self, idx):
        batch_files = self.files[idx * self.batch_size : (idx + 1) * self.batch_size]

        X_list, A_list, Y_list = [], [], []
        for fname in batch_files:
            base = os.path.basename(fname)
            try:
                file_parts = base.split("_")
                label_str = file_parts[1].split(".")[0]
                label = int(label_str)
            except:
                label = 0  # 預設值

            descriptors, A, coords = extract_descriptors_and_build_graph(
                fname,
                max_num_nodes=self.max_num_nodes,
                feature_dim=self.feature_dim,
                eta=self.eta,
                dk=self.dk,
                method=self.method
            )

            if descriptors is None or A is None:
                continue

            X_list.append(descriptors)
            A_list.append(A)
            Y_list.append(label)
        
        if len(X_list) == 0:
            # 空batch處理
            X_batch = np.zeros((0, self.max_num_nodes, self.feature_dim))
            A_batch = np.zeros((0, self.max_num_nodes, self.max_num_nodes))
            Y_batch = np.array([])
            return [X_batch, A_batch], Y_batch

        max_nodes = self.max_num_nodes
        X_batch = np.zeros((len(X_list), max_nodes, self.feature_dim))
        A_batch = np.zeros((len(X_list), max_nodes, max_nodes))
        Y_batch = np.array(Y_list)

        for i, (x, a) in enumerate(zip(X_list, A_list)):
            n = x.shape[0]
            X_batch[i, :n, :] = x
            A_batch[i, :n, :n] = a

        return [X_batch, A_batch], Y_batch
