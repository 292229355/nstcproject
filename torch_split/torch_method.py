import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import numpy as np
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data
import cv2
import dlib
from PIL import Image
from sklearn.preprocessing import StandardScaler
from transformers import (
    AutoImageProcessor,
    SuperPointForKeypointDetection,
)
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.loader import DataLoader as GeoDataLoader
from torch_geometric.nn import GATConv, global_mean_pool
from tqdm.auto import tqdm
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
# 設置設備
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 初始化處理器和模型
processor = AutoImageProcessor.from_pretrained("magic-leap-community/superpoint")
superpoint_model = SuperPointForKeypointDetection.from_pretrained(
    "magic-leap-community/superpoint"
)
superpoint_model.eval()
superpoint_model.to(device)

# Dlib 初始化
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

def extract_descriptors_and_build_graph2(
    img_pth,
    max_num_nodes=500,  # 限制节点数
    feature_dim=256,    # SuperPoint 的描述子维度 256
    k=5                 # KNN 中的 K 值
):
    img = cv2.imread(img_pth)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = detector(gray)
    if len(faces) == 0:
        print(f"No faces detected in image {img_pth}.")
        # 如果检测不到人脸，返回空图
        x = torch.zeros((1, feature_dim), dtype=torch.float)
        edge_index = torch.empty((2, 0), dtype=torch.long)
        y = torch.zeros((1, 2), dtype=torch.float)
        edge_attr = torch.empty((0, 1), dtype=torch.float)
        return x, edge_index, y, edge_attr

    face = faces[0]
    landmarks = predictor(gray, face)
    # 使用 dlib 提取人脸 68 个关键点
    points = np.array(
        [(landmarks.part(n).x, landmarks.part(n).y) for n in range(68)],
        dtype=np.int32,
    )

    mask = np.zeros_like(gray)
    cv2.fillConvexPoly(mask, points, 255)
    face_img = cv2.bitwise_and(gray, gray, mask=mask)

    # 将人脸改回 RGB
    face_img_pil = Image.fromarray(face_img).convert("RGB")

    # 使用 SuperPoint 进行特征提取
    inputs = processor(face_img_pil, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = superpoint_model(**inputs)

    image_mask = outputs.mask[0]
    image_indices = torch.nonzero(image_mask).squeeze()
    if image_indices.numel() == 0:
        print(f"No keypoints detected in image {img_pth}.")
        # 如果检测不到关键点，返回空图
        x = torch.zeros((1, feature_dim), dtype=torch.float)
        edge_index = torch.empty((2, 0), dtype=torch.long)
        y = torch.zeros((1, 2), dtype=torch.float)
        edge_attr = torch.empty((0, 1), dtype=torch.float)
        return x, edge_index, y, edge_attr

    # 获取关键点坐标和描述子
    keypoints = outputs.keypoints[0][image_indices]
    descriptors = outputs.descriptors[0][image_indices]

    keypoint_coords = keypoints.cpu().numpy()
    descriptors = descriptors.cpu().numpy()

    # 标准化描述子
    scaler = StandardScaler()
    descriptors = scaler.fit_transform(descriptors)

    # 限制节点数量
    if descriptors.shape[0] > max_num_nodes:
        indices = np.random.choice(descriptors.shape[0], max_num_nodes, replace=False)
        descriptors = descriptors[indices]
        keypoint_coords = keypoint_coords[indices]

    num_nodes = descriptors.shape[0]

    # 使用 KNN 构建图
    # 计算节点之间的欧氏距离矩阵
    dists = np.linalg.norm(
        keypoint_coords[:, np.newaxis, :] - keypoint_coords[np.newaxis, :, :],
        axis=2
    )
    # 将对角线（自身距离）设为无限大，避免选到自己
    np.fill_diagonal(dists, np.inf)

    edge_index = []
    edge_attr = []
    for i in range(num_nodes):
        # 找到每个节点的 K 个最近邻
        neighbor_indices = np.argsort(dists[i])[:k]
        for j in neighbor_indices:
            distance = dists[i, j]
            # 添加边（可选是否添加双向边）
            edge_index.append([i, j])
            edge_index.append([j, i])
            edge_attr.append([1 / (distance + 1e-5)])
            edge_attr.append([1 / (distance + 1e-5)])

    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_attr, dtype=torch.float)

    x = torch.from_numpy(descriptors).float().to(device)
    y = torch.from_numpy(keypoint_coords).float().to(device)  # 关键点坐标

    return x, edge_index, y, edge_attr



def stable_sigmoid(x):
    """
    Stable sigmoid function to handle potential overflow.
    """
    return torch.where(
        x < 0, torch.exp(x) / (1 + torch.exp(x)), 1 / (1 + torch.exp(-x))
    )


def scm(des1, des2):
    """
    Similarity Calculation Module (SCM) for calculating similarity between two descriptors.
    Cosine similarity scaled by the norm of des1.
    """
    dotproduct = torch.sum(des1 * des2, dim=1) / (
        torch.norm(des1, dim=1) * torch.norm(des2, dim=1) + 1e-8
    )  # 加 1e-8 防止除零
    x = dotproduct / (torch.norm(des1, dim=1) ** 0.5 + 1e-8)
    similarity = stable_sigmoid(x)
    return similarity


def andm(A, gamma, beta):
    """
    Adaptive Neighbour Discovery Module (ANDM) for pruning and thresholding the adjacency matrix.
    Args:
        A (torch.Tensor): Adjacency matrix (n x n)
        gamma (torch.Tensor): Learnable parameter gamma (n x 1)
        beta (torch.Tensor): Learnable parameter beta (n x 1)
    Returns:
        AN (torch.Tensor): Normalized adjacency matrix after ANDM (n x n)
    """
    n = A.size(0)

    # Calculate the adaptive threshold Ti for each node
    mean_A = torch.mean(A, dim=1, keepdim=True)  # (n x 1)
    Ti = gamma * mean_A + beta  # (n x 1)

    # Apply the threshold to prune the edges
    AT = torch.where(A > Ti, A, torch.zeros_like(A))  # (n x n)

    # Normalize the adjacency matrix using vectorized operations
    AN = F.softmax(AT, dim=1) * (AT > 0).float()  # (n x n)

    return AN


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

        # Compute attention scores
        scores = torch.matmul(Q, K.t()) / torch.sqrt(
            torch.tensor(self.dk, dtype=torch.float32).to(X.device)
        )  # (n x n)
        attention_scores = F.softmax(scores, dim=1)  # (n x n)

        # Compute Matt
        Matt = torch.sigmoid(torch.matmul(attention_scores, V))  # (n x dk)
        Matt = torch.matmul(Matt, torch.ones((self.dk, 1), device=X.device))  # (n x 1)
        Matt = Matt.squeeze(1)  # (n,)

        # Expand Matt to (n x n) by repeating
        Matt = Matt.unsqueeze(1).repeat(1, X.size(0))  # (n x n)

        return Matt


def calculate_adjacency_matrix_with_andm(X, eta=0.5, dk=64, attention_module=None):
    """
    Calculate and optimize adjacency matrix using SCM, SAM, and ANDM.
    Args:
        X (torch.Tensor): Node feature matrix (n x d)
        eta (float): Balance parameter for combining SCM and SAM
        dk (int): Dimensionality of attention keys in SAM
        attention_module (nn.Module): Instance of AttentionModule
    Returns:
        AN (torch.Tensor): Optimized adjacency matrix after ANDM (n x n)
    """
    n, d = X.size()

    # SCM: Calculate similarity matrix using cosine similarity
    with torch.no_grad():
        norm_X = F.normalize(X, p=2, dim=1)  # (n x d)
        Msim = torch.matmul(norm_X, norm_X.t())  # (n x n)
        Msim = torch.clamp(Msim, min=-1.0, max=1.0)  # 防止數值問題

        # Apply scm scaling
        # 在 PyTorch 中實現 scm 函數的向量化版本
        dotproduct = torch.sum(X.unsqueeze(1) * X.unsqueeze(0), dim=2) / (
            torch.norm(X, dim=1).unsqueeze(1) * torch.norm(X, dim=1).unsqueeze(0) + 1e-8
        )  # (n x n)
        x = dotproduct / (torch.norm(X, dim=1).unsqueeze(1) ** 0.5 + 1e-8)  # (n x n)
        scm_scores = stable_sigmoid(x)  # (n x n)

    # SAM: Self-attention mechanism using PyTorch
    if attention_module is not None:
        Matt = attention_module(X)  # (n x n)
    else:
        Matt = torch.zeros(n, n).to(X.device)

    # Combine SCM and SAM
    A = eta * scm_scores + (1 - eta) * Matt  # (n x n)

    # Define learnable parameters for ANDM (gamma and beta)
    # 這裡建議將 gamma 和 beta 作為 ANDM 的一部分，而不是每次計算時隨機初始化
    # 為簡單起見，我們暫時使用隨機初始化
    gamma = torch.rand(n, 1, device=X.device)  # (n x 1)
    beta = torch.rand(n, 1, device=X.device)  # (n x 1)

    # Apply ANDM
    AN = andm(A, gamma, beta)  # (n x n)

    return AN


def extract_descriptors_and_build_graph_with_andm(
    img_pth, max_num_nodes=500, feature_dim=256, eta=0.5, dk=64, attention_module=None
):
    """
    Extract features, build graph, and calculate adjacency matrix with ANDM.
    Args:
        img_pth (str): Path to the image file
        max_num_nodes (int): Maximum number of nodes
        feature_dim (int): Dimension of feature descriptors
        eta (float): Balance parameter for combining SCM and SAM
        dk (int): Dimensionality of attention keys in SAM
        attention_module (nn.Module): Instance of AttentionModule
    Returns:
        tuple: (x, edge_index, y, edge_attr, adjacency_matrix)
    """
    # Step 1: Extract node features using the provided function
    x, edge_index, y, edge_attr = extract_descriptors_and_build_graph2(
        img_pth, max_num_nodes=max_num_nodes, feature_dim=feature_dim
    )

    # If no nodes are detected, return empty results
    if x.size(0) == 0:
        print("No nodes detected, returning zero adjacency matrix.")
        return (
            x,
            edge_index,
            y,
            edge_attr,
            torch.zeros((1, 1), dtype=torch.float).to(device),
        )

    # Convert node features to PyTorch tensor (already in tensor form)
    feature_matrix = x  # (n x d)

    # Step 2: Calculate adjacency matrix using ANDM
    adjacency_matrix = calculate_adjacency_matrix_with_andm(
        feature_matrix, eta=eta, dk=dk, attention_module=attention_module
    )

    return x, edge_index, y, edge_attr, adjacency_matrix


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
    ):
        """
        Dataset for extracting graph-based descriptors from images.

        Args:
            path (str): Path to the directory containing image files.
            mode (str): Dataset mode ('train', 'val', or 'test').
            max_num_nodes (int): Maximum number of nodes in the graph.
            feature_dim (int): Dimensionality of the feature descriptors.
            eta (float): Balance parameter for combining SCM and SAM.
            dk (int): Dimensionality of attention keys in SAM.
            attention_module (nn.Module): Instance of AttentionModule.
        """
        super(DescriptorGraphDataset, self).__init__()
        self.path = path
        self.mode = mode
        self.max_num_nodes = max_num_nodes
        self.feature_dim = feature_dim
        self.eta = eta
        self.dk = dk
        self.attention_module = attention_module

        # Get list of image files
        self.files = sorted(
            [
                os.path.join(path, x)
                for x in os.listdir(path)
                if x.endswith(".jpg") or x.endswith(".png")
            ]
        )

        if len(self.files) == 0:
            raise FileNotFoundError(f"No image files found in {path}.")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        fname = self.files[idx]

        # Extract graph data and adjacency matrix
        data = extract_descriptors_and_build_graph_with_andm(
            fname,
            max_num_nodes=self.max_num_nodes,
            feature_dim=self.feature_dim,
            eta=self.eta,
            dk=self.dk,
            attention_module=self.attention_module,
        )

        # Unpack the data
        x, edge_index, pos, edge_attr, adjacency_matrix = data

        # Determine the label from the filename
        try:
            file_parts = os.path.basename(fname).split("_")
            label_str = file_parts[1].split(".")[0]
            label = int(label_str)
        except:
            label = -1  # Use -1 for missing or invalid labels

        # Convert adjacency matrix to PyTorch Geometric edge index and edge attributes
        if adjacency_matrix.size(0) > 1:  # Ensure adjacency matrix is not empty
            adj_indices = torch.nonzero(adjacency_matrix, as_tuple=False).t()
            adj_weights = adjacency_matrix[adj_indices[0], adj_indices[1]]
        else:
            adj_indices = torch.empty((2, 0), dtype=torch.long).to(device)
            adj_weights = torch.empty((0,), dtype=torch.float).to(device)

        # Create PyTorch Geometric data object
        graph_data = Data(
            x=x.to(device),
            edge_index=adj_indices,
            edge_attr=adj_weights,
            y=torch.tensor([label], dtype=torch.float),
            pos=pos,
        )

        return graph_data


class GATClassifier(nn.Module):
    def __init__(self, input_dim):
        super(GATClassifier, self).__init__()
        self.conv1 = GATConv(input_dim, 128, heads=4, concat=True, edge_dim=1)
        self.bn1 = nn.BatchNorm1d(128 * 4)
        self.conv2 = GATConv(128 * 4, 64, heads=4, concat=False, edge_dim=1)
        self.bn2 = nn.BatchNorm1d(64)
        self.fc1 = nn.Linear(64, 32)
        self.fc2 = nn.Linear(32, 1)
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x, edge_index, batch, edge_attr=None):
        if edge_attr is not None and edge_attr.size(0) == 0:
            edge_attr = None  # 如果沒有邊特徵，設置為 None
        x = self.conv1(x, edge_index, edge_attr=edge_attr)
        x = self.bn1(x)
        x = F.elu(x)
        x = self.conv2(x, edge_index, edge_attr=edge_attr)
        x = self.bn2(x)
        x = F.elu(x)
        x = global_mean_pool(x, batch)
        x = self.dropout(x)
        x = self.fc1(x)
        x = F.elu(x)
        x = self.fc2(x)
        return x  # 返回 logits


def train(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0
    total_acc = 0
    for batch in tqdm(loader, desc="Training"):
        batch = batch.to(device)
        optimizer.zero_grad()
        logits = model(
            batch.x, batch.edge_index, batch.batch, edge_attr=batch.edge_attr
        ).squeeze()
        loss = criterion(logits, batch.y)
        loss.backward()
        optimizer.step()
        preds = torch.sigmoid(logits)
        acc = ((preds > 0.5).float() == batch.y).float().mean()
        total_loss += loss.item()
        total_acc += acc.item()
    return total_loss / len(loader), total_acc / len(loader)


def validate(model, loader, criterion):
    model.eval()
    total_loss = 0
    total_acc = 0
    all_labels = []
    all_probs = []
    with torch.no_grad():
        for batch in tqdm(loader, desc="Validating"):
            batch = batch.to(device)
            logits = model(
                batch.x, batch.edge_index, batch.batch, edge_attr=batch.edge_attr
            ).squeeze()
            loss = criterion(logits, batch.y)
            preds = torch.sigmoid(logits)
            acc = ((preds > 0.5).float() == batch.y).float().mean()
            total_loss += loss.item()
            total_acc += acc.item()
            all_labels.extend(batch.y.cpu().numpy())
            all_probs.extend(preds.cpu().numpy())
    avg_loss = total_loss / len(loader)
    avg_acc = total_acc / len(loader)
    return avg_loss, avg_acc, all_labels, all_probs


if __name__ == "__main__":
    dataset_dir = "/content/drive/MyDrive/ADM_dataset"
    exp_name = "ADM"

    os.makedirs("models", exist_ok=True)

    # 初始化注意力模塊
    attention_module = AttentionModule(input_dim=256, dk=64).to(device)
    attention_module.eval()  # 如果不需要訓練 SAM，可以設置為 eval()

    train_set = DescriptorGraphDataset(
        os.path.join(dataset_dir, "train"),
        mode="train",
        attention_module=attention_module,
    )
    valid_set = DescriptorGraphDataset(
        os.path.join(dataset_dir, "valid"),
        mode="valid",
        attention_module=attention_module,
    )

    # DataLoader
    batch_size = 64  # 減少批次大小
    train_loader = GeoDataLoader(
        train_set, batch_size=batch_size, shuffle=True, num_workers=0
    )
    valid_loader = GeoDataLoader(
        valid_set, batch_size=batch_size, shuffle=False, num_workers=0
    )

    sample_data = None
    for data in train_set:
        if data is not None:
            sample_data = data
            break
    if sample_data is None:
        print("No valid data found in training set.")
        exit()
    input_dim = sample_data.num_node_features

    model = GATClassifier(input_dim).to(device)
    # Binary classification 使用 BCEWithLogitsLoss
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    n_epochs = 50  # 增加 epoch 數量以更好地訓練模型
    patience = 10
    best_acc = 0
    stale = 0
    train_losses, valid_losses, train_accuracies, valid_accuracies = [], [], [], []

    for epoch in range(n_epochs):
        train_loss, train_acc = train(model, train_loader, optimizer, criterion)
        valid_loss, valid_acc, valid_labels, valid_probs = validate(
            model, valid_loader, criterion
        )

        train_losses.append(train_loss)
        valid_losses.append(valid_loss)
        train_accuracies.append(train_acc)
        valid_accuracies.append(valid_acc)

        print(
            f"[Epoch {epoch + 1:03d}/{n_epochs:03d}] "
            f"Train Loss: {train_loss:.5f}, Train Acc: {train_acc:.5f} | "
            f"Valid Loss: {valid_loss:.5f}, Valid Acc: {valid_acc:.5f}"
        )

        scheduler.step()

        if valid_acc > best_acc:
            print(f"Best model found at epoch {epoch + 1}, saving model")
            torch.save(model.state_dict(), f"models/{exp_name}_best.ckpt")
            best_acc = valid_acc
            stale = 0
            # 用來繪製 ROC
            best_valid_labels = valid_labels
            best_valid_probs = valid_probs
        else:
            stale += 1
            if stale > patience:
                print(
                    f"No improvement in {patience} consecutive epochs, early stopping"
                )
                break

    test_set = DescriptorGraphDataset(
        os.path.join(dataset_dir, "test"),
        mode="test",
        attention_module=attention_module,
    )
    test_loader = GeoDataLoader(
        test_set, batch_size=batch_size, shuffle=False, num_workers=4
    )

    model_file = f"models/{exp_name}_best.ckpt"
    model_best = GATClassifier(input_dim).to(device)
    model_best.load_state_dict(
        torch.load(model_file, map_location=device)
    )  # 使用 map_location
    model_best.eval()

    prediction = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing"):
            batch = batch.to(device)
            logits = model_best(
                batch.x, batch.edge_index, batch.batch, edge_attr=batch.edge_attr
            ).squeeze()
            preds = torch.sigmoid(logits)
            prediction.extend((preds > 0.5).long().cpu().numpy())

    df = pd.DataFrame()
    df["Id"] = [str(i).zfill(4) for i in range(1, len(test_set) + 1)]
    df["Category"] = prediction
    df.to_csv("submission.csv", index=False)
    print("Prediction results saved to submission.csv")

    if "best_valid_labels" in locals() and "best_valid_probs" in locals():
        best_valid_labels = np.array(best_valid_labels)
        best_valid_probs = np.array(best_valid_probs)

        fpr, tpr, thresholds = roc_curve(best_valid_labels, best_valid_probs)
        roc_auc = auc(fpr, tpr)

        plt.figure()
        lw = 2
        plt.plot(
            fpr,
            tpr,
            color="darkorange",
            lw=lw,
            label="ROC curve (AUC = %0.4f)" % roc_auc,
        )
        plt.plot([0, 1], [0, 1], color="navy", lw=lw, linestyle="--")
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("Receiver Operating Characteristic (ROC)")
        plt.legend(loc="lower right")
        plt.show()
    else:
        print("No validation data available to compute ROC curve.")