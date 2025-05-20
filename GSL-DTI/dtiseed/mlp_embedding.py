import torch
import torch.nn as nn
import numpy as np

# =========================
# 설정
# =========================
output_dim = 512
hidden_dim = 1024

# =========================
# MLP 임베딩 모델 정의
# =========================
class MLPEmbedder(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.net(x)

# =========================
# MLP 임베딩 추출 함수
# =========================
def extract_embedding(data: np.ndarray, input_dim: int, save_path: str):
    X = torch.tensor(data.astype(np.float32))
    model = MLPEmbedder(input_dim)
    model.eval()
    with torch.no_grad():
        embedding = model(X)
    torch.save(embedding, save_path)
    print(f"✓ Saved embedding to {save_path}, shape: {embedding.shape}")

# =========================
# 데이터 불러오기 및 실행
# =========================
drug_sim = np.loadtxt("../data/heter/Similarity_Matrix_Drugs.txt")
prot_sim = np.loadtxt("../data/heter/Similarity_Matrix_Proteins.txt")

extract_embedding(drug_sim, input_dim=708, save_path="../data/heter/drug_embedding_mlp.pt")
extract_embedding(prot_sim, input_dim=1512, save_path="../data/heter/protein_embedding_mlp.pt")
