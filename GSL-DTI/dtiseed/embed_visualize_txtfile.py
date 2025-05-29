import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler  # 정규화 추가
import os

def load_features_from_txt(file_path):
    """
    .txt 파일에서 feature vector를 로드하는 함수
    각 줄이 하나의 벡터라고 가정
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"파일이 존재하지 않습니다: {file_path}")
    try:
        features = np.loadtxt(file_path)
    except Exception as e:
        raise ValueError(f"파일을 로드하는 중 오류 발생: {e}")
    
    return features

def normalize_features(features):
    """
    feature vector를 정규화 (zero-mean, unit-variance)
    """
    scaler = StandardScaler()
    normalized = scaler.fit_transform(features)
    return normalized

def visualize_pca_txt(file_path, labels=None, title="PCA Visualization"):
    features = load_features_from_txt(file_path)
    features = normalize_features(features)  # 정규화 추가

    pca = PCA(n_components=2)
    reduced = pca.fit_transform(features)

    plt.figure(figsize=(8, 6))
    if labels is not None:
        scatter = plt.scatter(reduced[:, 0], reduced[:, 1], c=labels, cmap='tab10', alpha=0.7)
        plt.legend(*scatter.legend_elements(), title="Classes")
    else:
        plt.scatter(reduced[:, 0], reduced[:, 1], alpha=0.7)
    plt.title(title)
    plt.xlabel("PCA1")
    plt.ylabel("PCA2")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def visualize_tsne_txt(file_path, labels=None, title="t-SNE Visualization", perplexity=30, n_iter=1000):
    features = load_features_from_txt(file_path)
    features = normalize_features(features)  # 정규화 추가

    tsne = TSNE(n_components=2, perplexity=perplexity, n_iter=n_iter, random_state=42)
    reduced = tsne.fit_transform(features)

    plt.figure(figsize=(8, 6))
    if labels is not None:
        scatter = plt.scatter(reduced[:, 0], reduced[:, 1], c=labels, cmap='tab10', alpha=0.7)
        plt.legend(*scatter.legend_elements(), title="Classes")
    else:
        plt.scatter(reduced[:, 0], reduced[:, 1], alpha=0.7)
    plt.title(title)
    plt.xlabel("t-SNE1")
    plt.ylabel("t-SNE2")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# 예시 사용
if __name__ == "__main__":
    file_path = "../data/heter/Similarity_Matrix_Drugs.txt"

    # 예시: 라벨이 있다면 여기에 정의
    # labels = np.array([...])
    labels = None

    #visualize_pca_txt(file_path, labels=labels, title="Protein PCA Visualization")
    visualize_tsne_txt(file_path, labels=labels, title="Protein t-SNE Visualization")
