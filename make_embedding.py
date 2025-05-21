import numpy as np
from sklearn.decomposition import PCA
import torch

# 입력 파일 경로
drug_sim = np.loadtxt("Similarity_Matrix_Drugs.txt")
prot_sim = np.loadtxt("Similarity_Matrix_Proteins.txt")

# PCA로 512차원 축소
pca_drug = PCA(n_components=512)
pca_prot = PCA(n_components=512)

drug_embed = pca_drug.fit_transform(drug_sim)
prot_embed = pca_prot.fit_transform(prot_sim)

# float32 Tensor로 변환
drug_tensor = torch.tensor(drug_embed, dtype=torch.float32)
prot_tensor = torch.tensor(prot_embed, dtype=torch.float32)

# 저장 (.pt)
torch.save(drug_tensor, "drug_similarity_512.pt")
torch.save(prot_tensor, "protein_similarity_512.pt")

print("✓ drug_similarity_512.pt / protein_similarity_512.pt 저장 완료")
