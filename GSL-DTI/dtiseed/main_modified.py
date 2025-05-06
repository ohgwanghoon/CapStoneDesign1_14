# main.py (Part 'a'만 실행하도록 수정한 버전)

from utilsdtiseed import *
# 필요한 모듈만 임포트
from modeltestdtiseed import HAN_DTI, init # GSLDTI 대신 HAN_DTI 임포트
import numpy as np
import torch
# import torch.nn.functional as F # 학습/손실 계산 안 하므로 주석 처리 가능
# import torch.nn as nn # 학습/손실 계산 안 하므로 주석 처리 가능
# from tqdm import tqdm # 학습 루프 없으므로 주석 처리 가능
# from sklearn.metrics import roc_auc_score, f1_score # 평가 안 하므로 주석 처리 가능
import warnings
import os
# from sklearn.metrics.pairwise import cosine_similarity as cos # 사용 안 함

warnings.filterwarnings("ignore")
seed = 47
args = setup(default_configure, seed)

# --- HAN_DTI 관련 하이퍼파라미터 설정 ---
# 원본 main.py의 값을 참고하거나, 필요한 값으로 설정
in_size_initial = 512  # 초기 랜덤 특징 벡터 차원 (또는 실제 특징 사용 시 해당 차원)
han_hidden_size = 256 # HAN 내부 GNN의 hidden 차원
han_out_size = 128     # 최종 Drug/Protein 표현 벡터 차원
dropout_han = 0      # HAN 모듈에 전달할 dropout 값 (HAN 내부에서 사용되지 않는다면 0)

args['device'] = "cuda:0" if torch.cuda.is_available() else "cpu"

# --- 사용할 데이터셋 선택 ---
dataset_name = "heter" # 또는 "Es" 등 원하는 데이터셋 이름

# --- 데이터 로딩 및 입력 준비 ---
print(f"Loading dataset: {dataset_name}")
# load_dataset은 utilsdtiseed의 함수 (수정된 load_hetero 등이 호출됨)
dtidata, graph, num, all_meta_paths = load_dataset(dataset_name)

# 초기 노드 특징 생성 또는 로딩
# 원본 main.py처럼 랜덤 특징 사용 예시
hd_in_size = in_size_initial
hp_in_size = in_size_initial
hd = torch.randn((num[0], hd_in_size)) # num[0] = num_drug
hp = torch.randn((num[1], hp_in_size)) # num[1] = num_protein
features_d = hd.to(args['device'])
features_p = hp.to(args['device'])
node_feature = [features_d, features_p] # HAN_DTI 입력으로 사용될 리스트

# 그래프 객체를 GPU로 이동 (이전에 추가한 코드)
if args['device'] != "cpu":
    print(f"Moving graphs to {args['device']}...")
    for i in range(len(graph)):
        graph[i] = graph[i].to(args['device'])
    print("Graphs moved.")

# --- Part 'a' 실행: HAN_DTI 모듈 실행 ---
print("Initializing HAN_DTI module...")
# GSLDTI 대신 HAN_DTI 객체 직접 생성
han_model = HAN_DTI(
    all_meta_paths=all_meta_paths,
    in_size=[hd_in_size, hp_in_size],            # 초기 특징 차원 리스트
    hidden_size=[han_hidden_size, han_hidden_size], # HAN 내부 hidden 차원 리스트
    out_size=[han_out_size, han_out_size],          # 최종 출력 차원 리스트
    dropout=dropout_han                         # Dropout 값
).to(args['device'])

print("Running HAN_DTI forward pass to get representations...")
han_model.eval() # 학습이 아니므로 평가 모드로 설정
with torch.no_grad(): # 그래디언트 계산 비활성화 (메모리 절약, 속도 향상)
    # HAN_DTI 모델 실행하여 Drug, Protein 표현 벡터 얻기
    drug_representation, protein_representation = han_model(graph, node_feature[0], node_feature[1])

# --- 결과 확인 ---
print("Part 'a' completed.")
print(f"Drug Representation Shape: {drug_representation.shape}")    # 결과: [num_drug, han_out_size]
print(f"Protein Representation Shape: {protein_representation.shape}") # 결과: [num_protein, han_out_size]

# --- 얻어진 표현 벡터 활용 ---
# 예를 들어, 파일로 저장하거나 다른 분석에 사용
print("Saving representations...")
torch.save(drug_representation, f'{dataset_name}_drug_repr_han.pt')
torch.save(protein_representation, f'{dataset_name}_protein_repr_han.pt')

print("Representations obtained. Script finished.")

# --- 기존 학습/평가 관련 함수 및 호출 부분 삭제 또는 주석 처리 ---
# def main(tr, te, seed): ...
# def train(...): ...
# def main_test(...): ...
# train_indeces, test_indeces = get_cross(dtidata)
# main(train_indeces, test_indeces, seed)