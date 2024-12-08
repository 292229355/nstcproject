import numpy as np
import tensorflow as tf

def andm(A, gamma, beta):
    n = A.shape[0]
    mean_A = np.mean(A, axis=1, keepdims=True)
    Ti = gamma * mean_A + beta
    AT = np.where(A > Ti, A, 0)

    AN = np.zeros_like(AT)
    for i in range(n):
        neighbors = np.where(AT[i] > 0)[0]
        if len(neighbors) > 0:
            AN[i, neighbors] = np.exp(AT[i, neighbors]) / np.sum(np.exp(AT[i, neighbors]))
    return AN

def sam_matrix(X, dk=64, seed=42):
    # 簡化計算 Q, S, O:
    # 使用固定隨機種子初始化，避免每次呼叫都不同
    np.random.seed(seed)
    n, d = X.shape

    # Q, S, O as numpy arrays to reduce overhead of new layers each time
    Q_w = np.random.randn(d, dk).astype(np.float32)
    S_w = np.random.randn(d, dk).astype(np.float32)
    O_w = np.random.randn(d, n).astype(np.float32)

    X_tensor = tf.convert_to_tensor(X, dtype=tf.float32)
    Q = tf.matmul(X_tensor, Q_w)
    S = tf.matmul(X_tensor, S_w)
    O = tf.matmul(X_tensor, O_w)

    attention_scores = tf.nn.softmax(tf.matmul(Q, S, transpose_b=True) / tf.sqrt(float(dk)))
    Matt = tf.sigmoid(tf.matmul(attention_scores, O)).numpy()
    return Matt

def andm_adjacency(X, similarity_matrix_func, eta=0.5, dk=64):
    # 計算相似度矩陣
    Msim = similarity_matrix_func(X)
    # 計算SAM矩陣
    Matt = sam_matrix(X, dk=dk)
    A = eta * Msim + (1 - eta) * Matt

    # 隨機 gamma, beta
    n = A.shape[0]
    gamma = np.random.rand(n, 1)
    beta = np.random.rand(n, 1)

    AN = andm(A, gamma, beta)
    return AN

def threshold_adjacency(sim_matrix, threshold=0.5):
    # 簡單設定一個閾值，如果相似度大於threshold就連邊
    A = (sim_matrix > threshold).astype(np.float32)
    return A

def knn_adjacency(sim_matrix, k=5):
    # 對每個節點選擇相似度最高的k個鄰居連邊
    n = sim_matrix.shape[0]
    A = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        # 排序並選擇 top k (不包含自己)
        neighbors = np.argsort(sim_matrix[i])[::-1]  # descending
        neighbors = neighbors[neighbors != i]
        top_k = neighbors[:k]
        A[i, top_k] = 1.0
    return A
