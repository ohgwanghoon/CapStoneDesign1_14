from utilsdtiseed import *
from modeltestdtiseed import HAN_DTI, MLP, init 
from modeltestdtiseed_part_a import DTI_Model
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, f1_score, auc
from sklearn.metrics import auc as auc_sklean
import warnings
import os
from sklearn.model_selection import StratifiedKFold

warnings.filterwarnings("ignore")
seed = 47
args = setup(default_configure, seed)

# --- HAN_DTI 관련 하이퍼파라미터 설정 ---
in_size_initial = 512  # 초기 랜덤 특징 벡터 차원 (또는 실제 특징 사용 시 해당 차원)
han_hidden_size = 256 # HAN 내부 GNN의 hidden 차원
han_out_size = 128     # 최종 Drug/Protein 표현 벡터 차원
dropout_han = 0.5      # HAN 모듈에 전달할 dropout 값 (HAN 내부에서 사용되지 않는다면 0)
learning_rate = 0.001  # 모델 학습률
weight_decay = 1e-5 # l2 정규화 가중치
epochs = 300           # 에포크 수
mlp_input_dim_val = han_out_size * 2 # drug 임베딩 + protein 임베딩 차원


args['device'] = "cuda:0" if torch.cuda.is_available() else "cpu"
save_dir = "../modelSave_part_a"
os.makedirs(save_dir, exist_ok=True)

# --- 사용할 데이터셋 선택 ---
dataset_name = "heter" # 또는 "Es" 등 원하는 데이터셋 이름

# --- 데이터 로딩 및 입력 준비 ---
print(f"Loading dataset: {dataset_name}")
# load_dataset은 utilsdtiseed의 함수 (수정된 load_hetero 등이 호출됨)
dtidata, graph_list, num_nodes, all_meta_paths = load_dataset(dataset_name)

# 초기 노드 특징 생성 또는 로딩
# 원본 main.py처럼 랜덤 특징 사용 예시
hd_in_size = in_size_initial
hp_in_size = in_size_initial
hd = torch.randn((num_nodes[0], hd_in_size)) # num[0] = num_drug
hp = torch.randn((num_nodes[1], hp_in_size)) # num[1] = num_protein
initial_node_features = [hd.to(args['device']), hp.to(args['device'])]

# 그래프 객체를 GPU로 이동 (이전에 추가한 코드)
if args['device'] != "cpu":
    print(f"Moving graphs to {args['device']}...")
    for i in range(len(graph_list)):
        graph_list[i] = graph_list[i].to(args['device'])
    print("Graphs moved.")

# 상호작용 쌍 인덱스
interaction_pairs = torch.tensor(dtidata[:, :2], dtype=torch.long).to(args['device'])
# 레이블
labels = torch.tensor(dtidata[:, 2], dtype=torch.long).to(args['device'])

# --- 손실 함수 정의 ---
# MLP의 출력이 LogSoftmax이므로 NLLLoss 사용
criterion = F.nll_loss

# --- 학습 및 평가 루프 (K-Fold) ---
data_indices = np.arange(dtidata.shape[0])
n_splits = 3 # Fold 수
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

fold_results = []

for fold, (train_idx_split, test_idx_split) in enumerate(skf.split(data_indices, dtidata[:, 2])):
    print(f"\n--- Fold {fold+1}/{n_splits} ---")

    # 현재 fold의 train/test용 상호작용 pair 인덱스 및 label 선택
    train_pairs_fold = interaction_pairs[train_idx_split]
    train_labels_fold = labels[train_idx_split]
    test_pairs_fold = interaction_pairs[test_idx_split]
    test_labels_fold = labels[test_idx_split]

    # 매 fold마다 모델 파라미터 초기화
    # 매번 새로 학습한다고 가정 -> 모델 다시 생성 및 옵티마이저 재생성
    print(f"Initializing DTI_Model for Fold {fold+1}...")
    model = DTI_Model(
        all_meta_paths=all_meta_paths,
        initial_feature_in_size=[hd_in_size, hp_in_size],
        han_hidden_size=[han_hidden_size, han_hidden_size],
        han_out_size=[han_out_size, han_out_size],
        mlp_input_dim=mlp_input_dim_val,
        dropout_han=dropout_han
    ).to(args['device'])

    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    best_test_roc = 0
    best_test_pr = 0

    for epoch in tqdm(range(epochs), desc=f"Fold {fold+1} Training"):
        model.train() # 학습 모드
        optimizer.zero_grad()

        # 모델 forward 호출 : 그래프, 초기 feature, 현재 학습 fold의 상호작용 pair 인덱스 전달
        outputs = model(graph_list, initial_node_features, train_pairs_fold)
        loss = criterion(outputs, train_labels_fold)

        loss.backward()
        optimizer.step()

        # 에포크마다 테스트 성능 평가 - 성능 개선 확인용
        if (epoch + 1) % 10 == 0: # 10 에포크마다 평가
            model.eval()
            with torch.no_grad():
                test_outputs = model(graph_list, initial_node_features, test_pairs_fold)
                test_loss = criterion(test_outputs, test_labels_fold)
                # ROC, PR 계산
                current_test_roc = get_roc(test_outputs, test_labels_fold)
                current_test_pr = get_pr(test_outputs, test_labels_fold)

                print(f"Epoch {epoch+1}: Test Loss: {test_loss.item():.4f}, Test ROC: {current_test_roc:.4f}, Test PR: {current_test_pr:.4f}")

                if current_test_pr > best_test_roc:
                    best_test_roc = current_test_roc
                    # 모델 저장
                    torch.save(model.state_dict(), os.path.join(save_dir, f"{dataset_name}_fold{fold+1}_best_roc_part_a.pt"))
                if current_test_pr > best_test_pr:
                    best_test_pr = current_test_pr


    # Fold 최종 평가
    model.eval()
    with torch.no_grad():
        test_outputs = model(graph_list, initial_node_features, test_pairs_fold)
        final_test_roc = get_roc(test_outputs, test_labels_fold)
        final_test_pr = get_pr(test_outputs, test_labels_fold)
        preds_class = test_outputs.argmax(dim=1)
        final_test_acc = (preds_class == test_labels_fold).float().mean().item()
        final_test_f1 = f1_score(test_labels_fold.cpu(), preds_class.cpu())
    
    print(f"Fold {fold+1} Results: ROC: {final_test_roc:.4f}, PR: {final_test_pr:.4f}, Acc: {final_test_acc:.4f}, F1: {final_test_f1:.4f}")
    print(f"Best ROC for Fold {fold+1} during epochs: {best_test_roc:.4f}")
    fold_results.append({'roc': final_test_roc, 'pr': final_test_pr, 'acc': final_test_acc, 'f1': final_test_f1, 'best_roc_epoch': best_test_roc})

# --- 교차 검증 결과 평균 ---
if fold_results:
    avg_roc = np.mean([res['roc'] for res in fold_results])
    avg_pr = np.mean([res['pr'] for res in fold_results])
    avg_acc = np.mean([res['acc'] for res in fold_results])
    avg_f1 = np.mean([res['f1'] for res in fold_results])
    avg_best_roc_epoch = np.mean([res['best_roc_epoch'] for res in fold_results])

    print("\n--- Average Cross-Validation Results ---")
    print(f"Average ROC: {avg_roc:.4f}")
    print(f"Average PR (AUPR): {avg_pr:.4f}")
    print(f"Average Accuracy: {avg_acc:.4f}")
    print(f"Average F1-score: {avg_f1:.4f}")
    print(f"Average Best ROC (within epochs): {avg_best_roc_epoch:.4f}")
else:
    print("No fold results to average.")

print("\nSimplified model training and evaluation finished.")