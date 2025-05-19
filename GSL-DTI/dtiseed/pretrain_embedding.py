import pandas as pd
from transformers import RobertaTokenizer, RobertaModel
import torch
import esm
import numpy as np
import os
import torch.nn as nn


# 약물 임베딩 추출 함수 (입력 데이터도 device로 이동)
# output shape: (number of drugs) * (768=hidden size)
def get_drug_embeddings(smiles_list, method="cls"):
    # 1. 장치 설정 (GPU가 있으면 GPU, 없으면 CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    # 사전학습된 ChemBERTa 모델 로딩 및 장치 이동
    tokenizer = RobertaTokenizer.from_pretrained("seyonec/ChemBERTa-zinc-base-v1")
    model = RobertaModel.from_pretrained("seyonec/ChemBERTa-zinc-base-v1")
    model.to(device)  # 모델을 GPU로 이동
    model.eval()

    inputs = tokenizer(smiles_list, return_tensors="pt", padding=True, truncation=True)
    # inputs의 모든 tensor를 device로 이동
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model(**inputs)

    token_embeddings = outputs.last_hidden_state  # (batch_size, seq_len, hidden_size)
    attention_mask = inputs["attention_mask"].unsqueeze(-1)  # (batch_size, seq_len, 1)

    if method == "cls":
        return token_embeddings[:, 0, :]  # (batch_size, hidden_size)
    elif method == "mean":
        summed = (token_embeddings * attention_mask).sum(dim=1)
        counted = attention_mask.sum(dim=1)
        return summed / counted  # (batch_size, hidden_size)
    else:
        raise ValueError("method must be 'cls' or 'mean'")

def get_drug_chemBERTa_embedding():
    # 1. 장치 설정 (GPU가 있으면 GPU, 없으면 CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 2. CSV 파일에서 SMILES 읽기
    csv_file_path = "../data/heter/drug_smiles.csv"  # 실제 경로 확인 필요
    df = pd.read_csv(csv_file_path)
    smiles_list = df['smiles'].tolist()

    # 3. 임베딩 계산
    drug_embeddings = get_drug_embeddings(smiles_list, method="cls")

    # 4. 임베딩을 텍스트 파일로 저장
    drug_embeddings_np = drug_embeddings.cpu().numpy()  # 반드시 CPU로 이동 후 numpy 변환

    # 5. 결과 파일 경로
    output_file = "../init_embeddings/drug_chemBERTa_embeddings.txt"
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w") as f:
        for emb in drug_embeddings_np:
            emb_str = " ".join(map(str, emb))
            f.write(emb_str + "\n")

    print(f"임베딩을 '{output_file}'에 저장했습니다.")

# 단백질 임베딩 추출 함수
# output shape: (number of proteins) * (모델에 따라 달라짐. 아래 참조)
# 주의!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# esm 코드는 기존의 라이브러리 환경에서 안돌아간다. 새로운 가상환경에서 더 높은 버전의 pytorch를 설치해야한다.
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
def embed_protein_sequences(input_file: str, output_file: str, batch_size: int = 2) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 모델 로딩
    model, alphabet = esm.pretrained.esm2_t33_650M_UR50D() # 어떤 모델 쓸지 입력!!!!!!!!!!!!!!!!!!!!!!!
    #아래는 모델에 따른 각 protein embedding 별 출력 feature의 수
    # esm2_t30_150M_UR50D: 640
    # esm2_t33_650M_UR50D: 1280
    # esm2_t36_3B_UR50D : 2560
    model = model.to(device)
    model.eval()
    batch_converter = alphabet.get_batch_converter()

    # 시퀀스 로딩
    with open(input_file, "r") as f:
        sequences = [line.strip() for line in f if line.strip()]
    print(f"Loaded {len(sequences)} sequences from {input_file}")

    max_length = 1024
    stride = 512
    embeddings = []

    def sliding_window(seq, window_size=max_length, stride=stride):
        return [seq[i:i + window_size] for i in range(0, len(seq), stride) if len(seq[i:i + window_size]) >= 10]

    num_layers = len(model.layers)  # 자동 레이어 수 감지

    for i in range(0, len(sequences), batch_size):
        batch_sequences = sequences[i:i + batch_size]
        for idx, seq in enumerate(batch_sequences):
            if not seq:
                print(f"[Warning] Sequence {i + idx + 1} is empty. Skipping.")
                continue

            if len(seq) <= max_length:
                batch_data = [(f"seq_{i + idx + 1}", seq)]
                _, _, tokens = batch_converter(batch_data)
                tokens = tokens.to(device)

                with torch.no_grad():
                    results = model(tokens, repr_layers=[num_layers], return_contacts=False)
                    rep = results["representations"][num_layers]
                    seq_len = (tokens[0] != alphabet.padding_idx).sum() - 2
                    mean_repr = rep[0, 1:seq_len + 1].mean(dim=0).cpu().numpy()
                    embeddings.append(mean_repr)
            else:
                windows = sliding_window(seq)
                window_reprs = []
                for w_idx, window_seq in enumerate(windows):
                    batch_data = [(f"seq_{i + idx + 1}_win{w_idx + 1}", window_seq)]
                    _, _, tokens = batch_converter(batch_data)
                    tokens = tokens.to(device)

                    with torch.no_grad():
                        results = model(tokens, repr_layers=[num_layers], return_contacts=False)
                        rep = results["representations"][num_layers]
                        seq_len = (tokens[0] != alphabet.padding_idx).sum() - 2
                        mean_repr = rep[0, 1:seq_len + 1].mean(dim=0).cpu().numpy()
                        window_reprs.append(mean_repr)

                if window_reprs:
                    final_repr = np.mean(window_reprs, axis=0)
                    embeddings.append(final_repr)
                else:
                    print(f"[Warning] Sequence {i + idx + 1} too long, but no valid windows. Skipping.")

        print(f"Processed batch {i // batch_size + 1}/{(len(sequences) + batch_size - 1) // batch_size}")

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w") as f:
        for emb in embeddings:
            emb_str = " ".join(map(str, emb))
            f.write(f"{emb_str}\n")

    print(f"Saved {len(embeddings)} embeddings to {output_file}")

#모델 임베딩 차원 조정(MLP 사용)
class FeatureAlignMLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dims=None):
        super(FeatureAlignMLP, self).__init__()
        
        if hidden_dims is None:
            hidden_dims = [max(input_dim, output_dim)]  # 기본: 입력 또는 출력 중 더 큰 차원 하나의 은닉층

        layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.ReLU())
            prev_dim = h_dim
        
        layers.append(nn.Linear(prev_dim, output_dim))  # 최종 출력 차원 맞추기

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

# 예시 실행; make protein initial embeddings
# prot_input_file = "../data/heter/protein_seq.txt"
# prot_output_file = "../init_embeddings/protein_esm_embeddings.txt"
# embed_protein_sequences(prot_input_file, prot_output_file)

# make drug initial embeddings
get_drug_chemBERTa_embedding()