import torch
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler  # 추가됨

def load_features_from_pt(file_path):
    """
    .pt 파일로부터 feature vector를 로드하는 함수
    """
    data = torch.load(file_path)
    if isinstance(data, dict):
        for key, value in data.items():
            if isinstance(value, torch.Tensor):
                features = value
                break
        else:
            raise ValueError("dict 안에 tensor 형식의 feature가 없습니다.")
    elif isinstance(data, torch.Tensor):
        features = data
    else:
        raise ValueError("지원하지 않는 .pt 파일 포맷입니다.")
    return features

def normalize_features(features):
    """
    feature vector를 정규화하는 함수
    """
    scaler = StandardScaler()
    normalized = scaler.fit_transform(features)
    return normalized

def visualize_with_pca(features, labels=None, title="PCA Visualization"):
    pca = PCA(n_components=2)
    reduced = pca.fit_transform(features)

    plt.figure(figsize=(8, 6))
    if labels is not None:
        scatter = plt.scatter(reduced[:, 0], reduced[:, 1], c=labels, cmap='tab10', alpha=0.7)
        plt.legend(*scatter.legend_elements(), title="Classes")
    else:
        plt.scatter(reduced[:, 0], reduced[:, 1], alpha=0.7)
    plt.title(title)
    plt.xlabel('PCA1')
    plt.ylabel('PCA2')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def visualize_with_tsne(features, labels=None, title="t-SNE Visualization", perplexity=30, n_iter=1000):
    tsne = TSNE(n_components=2, perplexity=perplexity, n_iter=n_iter, random_state=42)
    reduced = tsne.fit_transform(features)

    plt.figure(figsize=(8, 6))
    if labels is not None:
        scatter = plt.scatter(reduced[:, 0], reduced[:, 1], c=labels, cmap='tab10', alpha=0.7)
        plt.legend(*scatter.legend_elements(), title="Classes")
    else:
        plt.scatter(reduced[:, 0], reduced[:, 1], alpha=0.7)
    plt.title(title)
    plt.xlabel('t-SNE1')
    plt.ylabel('t-SNE2')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# 예시 사용
if __name__ == "__main__":
    file_path = "../init_feature/protein_embedding_autoencoder.pt"
    features = load_features_from_pt(file_path)
    features_np = features.detach().cpu().numpy()

    # 정규화 추가
    normalized_features = normalize_features(features_np)

    # label 정보가 있다면 아래처럼 전달
    # labels = ...
    labels = None  # 없으면 None으로 설정

    #visualize_with_pca(normalized_features, labels)
    visualize_with_tsne(normalized_features, labels)
