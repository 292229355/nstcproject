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
from extractfacefeature import extract_superpoint_features, extract_face_region

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
    """
    Stable sigmoid function to handle potential overflow.
    """
    return torch.where(
        x < 0, torch.exp(x) / (1 + torch.exp(x)), 1 / (1 + torch.exp(-x))
    )


def scm_pairwise(descriptors):
    """
    計算一組描述子間的兩兩相似度矩陣。
    descriptors: (n x d)
    回傳 (n x n) 的相似度矩陣
    """
    # des1 = des2 = descriptors
    # 計算 pairwise dot product
    dot_matrix = torch.matmul(descriptors, descriptors.t())  # (n x n)
    norm_vec = torch.norm(descriptors, dim=1)  # (n,)
    # 計算餘弦相似度矩陣
    cos_matrix = dot_matrix / (torch.ger(norm_vec, norm_vec) + 1e-8)  # (n x n)
    # x_matrix = cos_matrix / ((norm(des1)**0.5) + 1e-8)
    # 這裡 des1 是行方向節點，可用 norm_vec[i] 做為參考
    x_matrix = cos_matrix / (norm_vec.sqrt().unsqueeze(1) + 1e-8)  # (n x n)
    similarity_matrix = stable_sigmoid(x_matrix)
    return similarity_matrix


class AttentionModule(nn.Module):
    """
    Self-Attention Mechanism (SAM) implemented in PyTorch.
    """

    def __init__(self, input_dim, dk=64):
        super(AttentionModule, self).__init__()
        self.Q = nn.Linear(input_dim, dk, bias=False)
        self.K = nn.Linear(input_dim, dk, bias=False)
        self.V = nn.Linear(input_dim, dk, bias=False)
        self.dk = dk

    def forward(self, X):
        """
        Args:
            X (torch.Tensor): Node feature matrix (n x d)
        Returns:
            Matt (torch.Tensor): Attention-based adjacency matrix (n x n)
        """
        Q = self.Q(X)  # (n x dk)
        K = self.K(X)  # (n x dk)
        V = self.V(X)  # (n x dk)

        scores = torch.matmul(Q, K.t()) / torch.sqrt(
            torch.tensor(self.dk, dtype=torch.float32).to(X.device)
        )  # (n x n)
        attention_scores = F.softmax(scores, dim=1)  # (n x n)

        Matt = torch.sigmoid(torch.matmul(attention_scores, V))  # (n x dk)
        Matt = torch.matmul(Matt, torch.ones((self.dk, 1), device=X.device))  # (n x 1)
        Matt = Matt.squeeze(1)  # (n,)
        Matt = Matt.unsqueeze(1).repeat(1, X.size(0))  # (n x n)

        return Matt

def filter_descriptors_adaptively(descriptors, device, method="similarity", retain_ratio=0.8):
    """
    根據描述子特性動態篩選描述子。
    Args:
        descriptors (torch.Tensor): (n x d) 的描述子張量。
        device: 運算裝置。
        method (str): "similarity" 或 "norm"。
        retain_ratio (float): 要保留的比例（例如0.8代表保留80%描述子）。

    Returns:
        filtered_descriptors (torch.Tensor): 篩選後的描述子 (m x d)。
        selected_indices (torch.Tensor): 被保留的描述子索引。
    """

    n = descriptors.size(0)
    if n == 0:
        return descriptors, torch.arange(n)

    if method == "similarity":
        # 計算所有描述子間相似度
        similarity_matrix = scm_pairwise(descriptors)
        # 計算每個描述子與其他描述子的平均相似度
        avg_sim = similarity_matrix.mean(dim=1)  # (n,)
        score = avg_sim
    else:
        # 使用描述子 norm 作為品質依據
        desc_norm = torch.norm(descriptors, dim=1)  # (n,)
        score = desc_norm

    # 根據 score 將描述子排序，保留 score 高的 top retain_ratio
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
    """
    基於臉部特徵與 SuperPoint 特徵提取，直接透過 SCM 與 AttentionModule
    計算得到的關聯矩陣產生圖。

    Args:
        img_pth (str): 圖像路徑。
        processor: SuperPoint 的處理器。
        superpoint_model: SuperPoint 模型。
        device: 計算設備 (CPU 或 GPU)。
        max_num_nodes (int): 最大節點數量。
        feature_dim (int): 描述子維度。
        eta (float): SCM與SAM組合的平衡參數。
        attention_module (nn.Module): AttentionModule 實例。
        threshold (float): 用於篩選邊的閾值。

    Returns:
        x (torch.Tensor): 節點特徵 (n x d)。
        edge_index (torch.Tensor): 邊的索引 (2 x m)。
        pos (torch.Tensor): 節點的座標 (n x 2)。
        edge_attr (torch.Tensor): 邊的權重 (m,)。
        adjacency_matrix (torch.Tensor): (n x n) 的關聯矩陣。

    整合臉部提取、特徵提取、自適應篩選描述子、以及基於相似度與注意力機制的圖建構。
    """

    face_img_pil, face_landmarks = extract_face_region(img_pth)
    if face_img_pil is None:
        print(f"No faces detected in image {img_pth}.")
        # 回傳空圖
        x = torch.zeros((1, feature_dim), dtype=torch.float).to(device)
        edge_index = torch.empty((2, 0), dtype=torch.long).to(device)
        pos = torch.zeros((1, 2), dtype=torch.float).to(device)
        edge_attr = torch.empty((0,), dtype=torch.float).to(device)
        adjacency_matrix = torch.zeros((1, 1), dtype=torch.float).to(device)
        return x, edge_index, pos, edge_attr, adjacency_matrix

    descriptors, keypoint_coords = extract_superpoint_features(
        face_img_pil, processor, superpoint_model, device, max_num_nodes, feature_dim
    )
    if descriptors is None or descriptors.shape[0] == 0:
        print(f"No keypoints detected in image {img_pth}.")
        # 回傳空圖
        x = torch.zeros((1, feature_dim), dtype=torch.float).to(device)
        edge_index = torch.empty((2, 0), dtype=torch.long).to(device)
        pos = torch.zeros((1, 2), dtype=torch.float).to(device)
        edge_attr = torch.empty((0,), dtype=torch.float).to(device)
        adjacency_matrix = torch.zeros((1, 1), dtype=torch.float).to(device)
        return x, edge_index, pos, edge_attr, adjacency_matrix

    # 將描述子轉換為張量
    x = torch.from_numpy(descriptors).float().to(device)  # (n x d)
    pos = torch.from_numpy(keypoint_coords).float().to(device)  # (n x 2)

    # 自適應篩選描述子
    x_filtered, selected_indices = filter_descriptors_adaptively(
        x, device, method=adaptive_filter_method, retain_ratio=adaptive_retain_ratio
    )
    pos_filtered = pos[selected_indices]

    n = x_filtered.size(0)
    if n < 2:
        # 若篩選後節點不足，不足以構建有意義的圖
        adjacency_matrix = torch.zeros((n, n), dtype=torch.float).to(device)
        edge_index = torch.empty((2, 0), dtype=torch.long).to(device)
        edge_attr = torch.empty((0,), dtype=torch.float).to(device)
        return x_filtered, edge_index, pos_filtered, edge_attr, adjacency_matrix

    # 計算 SCM 相似度矩陣 (n x n)
    similarity_matrix = scm_pairwise(x_filtered)

    # 使用注意力模組計算 (n x n) attention matrix
    if attention_module is not None:
        attention_matrix = attention_module(x_filtered)
    else:
        attention_matrix = similarity_matrix

    # 組合相似度與注意力矩陣
    adjacency_matrix = eta * similarity_matrix + (1 - eta) * attention_matrix

    # 使用 threshold 篩選邊
    mask = adjacency_matrix >= threshold
    edge_indices = mask.nonzero(as_tuple=False).t()  # (2, m)
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

        # Get list of image files
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

        # 從檔名中取出標籤（如有需要）
        try:
            file_parts = os.path.basename(fname).split("_")
            label_str = file_parts[1].split(".")[0]
            label = int(label_str)
        except:
            label = -1

        # 建立 PyTorch Geometric Data
        graph_data = Data(
            x=x.to(device),
            edge_index=edge_index.to(device),
            edge_attr=edge_attr.to(device),
            y=torch.tensor([label], dtype=torch.float, device=device),
            pos=pos.to(device),
        )

        return graph_data
