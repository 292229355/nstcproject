import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data
from transformers import (
    AutoImageProcessor,
    SuperPointForKeypointDetection,
)
import dlib
import torch.nn as nn
import torch.nn.functional as F
import os
import cv2
import numpy as np
from PIL import Image
from sklearn.preprocessing import StandardScaler
from extractfacefeature import extract_face_region

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

processor = AutoImageProcessor.from_pretrained("magic-leap-community/superpoint")
superpoint_model = SuperPointForKeypointDetection.from_pretrained(
    "magic-leap-community/superpoint"
)

superpoint_model.eval()
superpoint_model.to(device)

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

def stable_sigmoid(x):
    return torch.where(
        x < 0, torch.exp(x) / (1 + torch.exp(x)), 1 / (1 + torch.exp(-x))
    )

def scm_pairwise(descriptors):
    dot_matrix = torch.matmul(descriptors, descriptors.t())  # (n x n)
    norm_vec = torch.norm(descriptors, dim=1)  # (n,)
    cos_matrix = dot_matrix / (torch.ger(norm_vec, norm_vec) + 1e-8)  # (n x n)
    x_matrix = cos_matrix / (norm_vec.sqrt().unsqueeze(1) + 1e-8)  # (n x n)
    similarity_matrix = stable_sigmoid(x_matrix)
    return similarity_matrix

class AttentionModule(nn.Module):
    def __init__(self, input_dim, dk=64):
        super(AttentionModule, self).__init__()
        self.Q = nn.Linear(input_dim, dk, bias=False)
        self.K = nn.Linear(input_dim, dk, bias=False)
        self.V = nn.Linear(input_dim, dk, bias=False)
        self.dk = dk

    def forward(self, X):
        Q = self.Q(X)
        K = self.K(X)
        V = self.V(X)
        scores = torch.matmul(Q, K.t()) / torch.sqrt(
            torch.tensor(self.dk, dtype=torch.float32).to(X.device)
        )
        attention_scores = F.softmax(scores, dim=1)
        Matt = torch.sigmoid(torch.matmul(attention_scores, V))
        Matt = torch.matmul(Matt, torch.ones((self.dk, 1), device=X.device))
        Matt = Matt.squeeze(1)
        Matt = Matt.unsqueeze(1).repeat(1, X.size(0))
        return Matt



def filter_descriptors_adaptively(descriptors, device, method="similarity", retain_ratio=0.8):
    n = descriptors.size(0)
    if n == 0:
        return descriptors, torch.arange(n)

    if method == "similarity":
        similarity_matrix = scm_pairwise(descriptors)
        avg_sim = similarity_matrix.mean(dim=1)  # (n,)
        score = avg_sim
    else:
        desc_norm = torch.norm(descriptors, dim=1)
        score = desc_norm

    threshold_index = int(n * retain_ratio)
    sorted_vals, sorted_indices = torch.sort(score, descending=True)
    selected_indices = sorted_indices[:threshold_index]
    filtered_descriptors = descriptors[selected_indices]

    return filtered_descriptors, selected_indices

def extract_descriptors_and_build_graph2(
    img_pth,
    processor,
    superpoint_model,
    device,
    max_num_nodes=500,
    feature_dim=256,
    eta=0.5,
    attention_module=None,
    threshold=0.5,
    adaptive_filter_method="similarity",
    adaptive_retain_ratio=0.8
):
    # 現在 extract_face_region 已經同時取得八個區域的描述子與座標
    # 不再需要單獨呼叫 extract_superpoint_features
    combined_descriptors, combined_keypoints = extract_face_region(
        img_pth, processor, superpoint_model, device, max_num_nodes, feature_dim
    )

    if combined_descriptors is None or combined_keypoints is None:
        print(f"No faces or no keypoints in image {img_pth}.")
        x = torch.zeros((1, feature_dim), dtype=torch.float).to(device)
        edge_index = torch.empty((2, 0), dtype=torch.long).to(device)
        pos = torch.zeros((1, 2), dtype=torch.float).to(device)
        edge_attr = torch.empty((0,), dtype=torch.float).to(device)
        adjacency_matrix = torch.zeros((1, 1), dtype=torch.float).to(device)
        return x, edge_index, pos, edge_attr, adjacency_matrix

    x = torch.from_numpy(combined_descriptors).float().to(device)
    pos = torch.from_numpy(combined_keypoints).float().to(device)

    # 自適應篩選
    x_filtered, selected_indices = filter_descriptors_adaptively(
        x, device, method=adaptive_filter_method, retain_ratio=adaptive_retain_ratio
    )
    pos_filtered = pos[selected_indices]

    n = x_filtered.size(0)
    if n < 2:
        adjacency_matrix = torch.zeros((n, n), dtype=torch.float).to(device)
        edge_index = torch.empty((2, 0), dtype=torch.long).to(device)
        edge_attr = torch.empty((0,), dtype=torch.float).to(device)
        return x_filtered, edge_index, pos_filtered, edge_attr, adjacency_matrix

    # 計算 SCM 相似度矩陣
    similarity_matrix = scm_pairwise(x_filtered)

    # 使用注意力模組
    if attention_module is not None:
        attention_matrix = attention_module(x_filtered)
    else:
        attention_matrix = similarity_matrix

    # 組合
    adjacency_matrix = eta * similarity_matrix + (1 - eta) * attention_matrix

    # 篩選邊
    mask = adjacency_matrix >= threshold
    edge_indices = mask.nonzero(as_tuple=False).t()
    edge_weights = adjacency_matrix[edge_indices[0], edge_indices[1]]

    return x_filtered, edge_indices, pos_filtered, edge_weights, adjacency_matrix


class DescriptorGraphDataset(Dataset):
    def __init__(
        self,
        path,
        mode="train",
        max_num_nodes=500,
        feature_dim=256,
        eta=0.5,
        dk=64,
        attention_module=None,
        threshold=0.5
    ):
        super(DescriptorGraphDataset, self).__init__()
        self.path = path
        self.mode = mode
        self.max_num_nodes = max_num_nodes
        self.feature_dim = feature_dim
        self.eta = eta
        self.dk = dk
        self.attention_module = attention_module
        self.threshold = threshold

        self.files = sorted(
            [
                os.path.join(path, x)
                for x in os.listdir(path)
                if x.lower().endswith(".jpg") or x.lower().endswith(".png")
            ]
        )

        if len(self.files) == 0:
            raise FileNotFoundError(f"No image files found in {path}.")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        fname = self.files[idx]
        data = extract_descriptors_and_build_graph2(
            fname,
            processor,
            superpoint_model,
            device,
            max_num_nodes=self.max_num_nodes,
            feature_dim=self.feature_dim,
            eta=self.eta,
            attention_module=self.attention_module,
            threshold=self.threshold
        )

        x, edge_index, pos, edge_attr, adjacency_matrix = data

        # 從檔名推測標籤
        try:
            file_parts = os.path.basename(fname).split("_")
            label_str = file_parts[1].split(".")[0]
            label = int(label_str)
        except:
            label = -1

        graph_data = Data(
            x=x.to(device),
            edge_index=edge_index.to(device),
            edge_attr=edge_attr.to(device),
            y=torch.tensor([label], dtype=torch.float, device=device),
            pos=pos.to(device),
        )

        return graph_data
