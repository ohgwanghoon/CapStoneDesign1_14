import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# =========================
# 공통 설정
# =========================
output_dim = 512
hidden_dim = 1024
epochs = 200
lr = 1e-3

# =========================
# Autoencoder 클래스 정의
# =========================
class Autoencoder(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(output_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )

    def forward(self, x):
        z = self.encoder(x)
        out = self.decoder(z)
        return out, z

def train_autoencoder(data: np.ndarray, input_dim: int, save_path: str):
    X = torch.tensor(data.astype(np.float32))
    model = Autoencoder(input_dim)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    for _ in range(epochs):
        model.train()
        optimizer.zero_grad()
        reconstructed, _ = model(X)
        loss = criterion(reconstructed, X)
        loss.backward()
        optimizer.step()

    embedding = model.encoder(X).detach()
    torch.save(embedding, save_path)

# =========================
# 데이터 불러오기 및 실행
# =========================
drug_sim = np.loadtxt("../data/heter/Similarity_Matrix_Drugs.txt")
prot_sim = np.loadtxt("../data/heter/Similarity_Matrix_Proteins.txt")

train_autoencoder(drug_sim, input_dim=708, save_path="../data/heter/drug_embedding_autoencoder.pt")
train_autoencoder(prot_sim, input_dim=1512, save_path="../data/heter/protein_embedding_autoencoder.pt")
