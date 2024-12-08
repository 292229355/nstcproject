import numpy as np

def stable_sigmoid(x):
    return 1 / (1 + np.exp(-x))

def scm(des1, des2):
    # SCM為原始定義的特殊相似度
    dotproduct = np.dot(des1, des2) / (np.linalg.norm(des1)*np.linalg.norm(des2) + 1e-8)
    x = dotproduct / (np.linalg.norm(des1)**0.5 + 1e-8)
    similarity = stable_sigmoid(x)
    return similarity

def cosine_similarity_matrix(X):
    # X shape: (n, d)
    # 向量化Cosine Similarity計算
    norms = np.linalg.norm(X, axis=1, keepdims=True) + 1e-8
    X_normed = X / norms
    return np.dot(X_normed, X_normed.T)

def scm_similarity_matrix(X):
    # 向量化 SCM 計算
    # 首先計算 dot product
    norms = np.linalg.norm(X, axis=1, keepdims=True) + 1e-8
    X_normed = X / norms
    dot_matrix = X_normed @ X_normed.T

    # 計算 denominator 用於 SCM
    half_norms = (norms.squeeze()**0.5 + 1e-8)
    denom = half_norms.reshape(-1,1)

    return stable_sigmoid(dot_matrix / denom)

def pearson_correlation_matrix(X):
    # Pearson correlation 作為相似度
    # 對 X 沿各特徵列中心化
    X_mean = np.mean(X, axis=1, keepdims=True)
    X_centered = X - X_mean
    norms = np.linalg.norm(X_centered, axis=1, keepdims=True) + 1e-8
    X_normed = X_centered / norms
    return np.dot(X_normed, X_normed.T)
