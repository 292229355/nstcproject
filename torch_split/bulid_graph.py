import torch
import numpy as np
def build_knn_graph(descriptors, keypoint_coords, k=5, device='cpu'):
    """
    使用 KNN 建立圖的邊和邊屬性。

    Args:
        descriptors (np.ndarray): 特徵描述子。
        keypoint_coords (np.ndarray): 特徵點座標。
        k (int): 每個節點連接的最近鄰數量。
        device (str or torch.device): 計算設備。

    Returns:
        x (torch.Tensor): 節點特徵。
        edge_index (torch.Tensor): 邊的索引。
        y (torch.Tensor): 節點座標。
        edge_attr (torch.Tensor): 邊屬性。
    """
    if descriptors is None or keypoint_coords is None:
        # 返回空的圖結構
        x = torch.zeros((1, descriptors.shape[1] if descriptors is not None else 256), dtype=torch.float).to(device)
        edge_index = torch.empty((2, 0), dtype=torch.long).to(device)
        y = torch.zeros((1, 2), dtype=torch.float).to(device)
        edge_attr = torch.empty((0, 1), dtype=torch.float).to(device)
        return x, edge_index, y, edge_attr

    num_nodes = descriptors.shape[0]

    # 計算歐氏距離矩陣
    dists = np.linalg.norm(
        keypoint_coords[:, np.newaxis, :] - keypoint_coords[np.newaxis, :, :],
        axis=2
    )
    np.fill_diagonal(dists, np.inf)

    edge_index = []
    edge_attr = []
    for i in range(num_nodes):
        neighbor_indices = np.argsort(dists[i])[:k]
        for j in neighbor_indices:
            distance = dists[i, j]
            edge_index.append([i, j])
            edge_index.append([j, i])
            edge_attr.append([1 / (distance + 1e-5)])
            edge_attr.append([1 / (distance + 1e-5)])

    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous().to(device)
    edge_attr = torch.tensor(edge_attr, dtype=torch.float).to(device)

    x = torch.from_numpy(descriptors).float().to(device)
    y = torch.from_numpy(keypoint_coords).float().to(device)

    return x, edge_index, y, edge_attr
