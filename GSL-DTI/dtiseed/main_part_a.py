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
import pickle

warnings.filterwarnings("ignore")
seed = 47
args = setup(default_configure, seed)

# --- HAN_DTI 관련 하이퍼파라미터 설정 ---
in_size_initial = 512  # 초기 랜덤 특징 벡터 차원 (또는 실제 특징 사용 시 해당 차원)
han_hidden_size = 256 # HAN 내부 GNN의 hidden 차원
han_out_size = 128     # 최종 Drug/Protein 표현 벡터 차원
dropout_han = 0.5      # HAN 모듈에 전달할 dropout 값 (HAN 내부에서 사용되지 않는다면 0)
learning_rate = 1e-5  # 모델 학습률
weight_decay = 1e-4 # l2 정규화 가중치
epochs = 1000          # 에포크 수
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

# 초기 임베딩 플래그
init_feature_flag = 0

if init_feature_flag == 0 :
    hd = torch.randn((num_nodes[0], hd_in_size)) # num_nodes[0] = num_drug
    hp = torch.randn((num_nodes[1], hp_in_size)) # num_nodes[1] = num_protein
elif init_feature_flag == 1:
    # TODO Pretrained 사용
    # torch.Tensor 형태로 받아야 함!
    hd = torch.zeros((num_nodes[0], hd_in_size))
    hp = torch.zeros((num_nodes[1], hp_in_size))
    print("Pretrained")
elif init_feature_flag == 2:
    # TODO Similarity 사용
    # torch.Tensor 형태로 받아야 함!
    hd = torch.zeros((num_nodes[0], hd_in_size))
    hp = torch.zeros((num_nodes[1], hp_in_size))
    print("Similarity")


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

# 모든 fold의 결과를 저장할 딕셔너리
all_folds_history = {} # key : fold 번호, value : 해당 fold의 epoch 별 기록 딕셔너리

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

    fold_history = {
        'epochs': [],
        'train_loss': [],
        'train_acc': [],
        'test_loss': [],
        'test_roc': [],
        'test_acc': [],
        'test_pr' : [],
        # ROC/PR 커브를 위한 데이터 (매 평가 시점마다 저장)
        'test_true_labels': [], # 실제 테스트 레이블 리스트
        'test_pred_scores': [], # 모델의 예측 확률(점수) 리스트 (상호작용한다에 대한)
    }

    best_test_roc = 0
    best_test_pr = 0

    for epoch in tqdm(range(epochs), desc=f"Fold {fold+1} Training"):
        model.train() # 학습 모드
        optimizer.zero_grad()

        # 모델 forward 호출 : 그래프, 초기 feature, 현재 학습 fold의 상호작용 pair 인덱스 전달
        train_outputs = model(graph_list, initial_node_features, train_pairs_fold)
        current_train_loss = criterion(train_outputs, train_labels_fold)

        current_train_loss.backward()
        optimizer.step()

        # 에포크마다 테스트 성능 평가 - 성능 개선 확인용
        if (epoch + 1) % 10 == 0: # 10 에포크마다 평가
            model.eval()
            with torch.no_grad():
                # train accuracy 계산
                train_pred = train_outputs.argmax(dim=1)
                current_train_acc = (train_pred == train_labels_fold).float().mean().item()

                fold_history['epochs'].append(epoch + 1)
                fold_history['train_loss'].append(current_train_loss.item())
                fold_history['train_acc'].append(current_train_acc)

                # test output, loss 계산
                test_outputs = model(graph_list, initial_node_features, test_pairs_fold)
                current_test_loss = criterion(test_outputs, test_labels_fold)
                # ROC, PR 계산
                current_test_roc = get_roc(test_outputs, test_labels_fold)
                current_test_pr = get_pr(test_outputs, test_labels_fold)

                fold_history['test_loss'].append(current_test_loss.item())
                fold_history['test_roc'].append(current_test_roc)
                fold_history['test_pr'].append(current_test_pr)

                probabilities_class1 = torch.exp(test_outputs[:, 1])
                fold_history['test_true_labels'].append(test_labels_fold.cpu().numpy())
                fold_history['test_pred_scores'].append(probabilities_class1.cpu().numpy())

                print(f"\nEpoch {epoch+1}: Train Acc: {current_train_acc:.4f}, Test Loss: {current_test_loss.item():.4f}, Test ROC: {current_test_roc:.4f}, Test PR: {current_test_pr:.4f}")

                if current_test_roc > best_test_roc:
                    best_test_roc = current_test_roc
                    # 모델 저장
                    torch.save(model.state_dict(), os.path.join(save_dir, f"{dataset_name}_fold{fold+1}_best_roc_part_a.pt"))
                if current_test_pr > best_test_pr:
                    best_test_pr = current_test_pr

    all_folds_history[f"fold_{fold+1}"] = fold_history

    # Fold 최종 평가
    # model.eval()
    # with torch.no_grad():
    #     test_outputs = model(graph_list, initial_node_features, test_pairs_fold)
    #     final_test_roc = get_roc(test_outputs, test_labels_fold)
    #     final_test_pr = get_pr(test_outputs, test_labels_fold)
    #     preds_class = test_outputs.argmax(dim=1)
    #     final_test_acc = (preds_class == test_labels_fold).float().mean().item()
    #     final_test_f1 = f1_score(test_labels_fold.cpu(), preds_class.cpu())
    if fold_history['epochs']:
        final_test_roc = fold_history['test_roc'][-1]
        final_test_pr = fold_history['test_pr'][-1]

        last_eval_true_labels = fold_history['test_true_labels'][-1]
        last_eval_pred_scores = fold_history['test_pred_scores'][-1]

        threshold = 0.5
        last_eval_pred_classes = (last_eval_pred_scores >= threshold).astype(int)

        final_test_acc = np.mean(last_eval_pred_classes == last_eval_true_labels)
        final_test_f1 = f1_score(last_eval_true_labels, last_eval_pred_classes, zero_division=0) # zero_division 추가
    else: # 에폭 수가 10 미만이라 기록이 없을 경우 대비
            final_test_roc, final_test_pr, final_test_acc, final_test_f1 = 0, 0, 0, 0
            print("Warning: No evaluation metrics recorded for this fold (epochs < 10?).")
        
    print(f"Fold {fold+1} Results: ROC: {final_test_roc:.4f}, PR: {final_test_pr:.4f}, Acc: {final_test_acc:.4f}, F1: {final_test_f1:.4f}")
    print(f"Best ROC for Fold {fold+1} during epochs: {best_test_roc:.4f}")

    fold_results.append({
        'roc': final_test_roc,
        'pr': final_test_pr,
        'acc': final_test_acc,
        'f1': final_test_f1,
        'best_test_roc': best_test_roc
    })


history_file_path = os.path.join(save_dir, f"{dataset_name}_all_folds_history.pkl")
try:
    with open(history_file_path, 'wb') as f:
        pickle.dump(all_folds_history, f)
    print(f"\nAll folds history saved to: {history_file_path}")
except Exception as e:
    print("Error saving history file: {e}")

# --- 교차 검증 결과 평균 ---
# if fold_results:
#     avg_roc = np.mean([res['roc'] for res in fold_results])
#     avg_pr = np.mean([res['pr'] for res in fold_results])
#     avg_acc = np.mean([res['acc'] for res in fold_results])
#     avg_f1 = np.mean([res['f1'] for res in fold_results])
#     avg_best_roc_epoch = np.mean([res['best_roc_epoch'] for res in fold_results])

if all_folds_history:
    num_folds = len(all_folds_history)
    avg_final_roc = np.mean([all_folds_history[f"fold_{f+1}"]['test_roc'][-1] for f in range(num_folds) if all_folds_history[f"fold_{f+1}"]['test_roc']])
    avg_final_pr = np.mean([all_folds_history[f"fold_{f+1}"]['test_pr'][-1] for f in range(num_folds) if all_folds_history[f"fold_{f+1}"]['test_pr']])

    if fold_results:
        avg_final_acc = np.mean([res['acc'] for res in fold_results])
        avg_final_f1 = np.mean([res['f1'] for res in fold_results])
        avg_best_roc_in_fold = np.mean([res['best_test_roc'] for res in fold_results])
    else: # fold_results가 비었다면 (예: 에폭이 매우 적어 결과가 안 쌓임)
        avg_final_acc, avg_final_f1, avg_best_roc_in_fold = 0, 0, 0

    print("\n--- Average Cross-Validation Results ---")
    print(f"Average ROC: {avg_final_roc:.4f}")
    print(f"Average PR (AUPR): {avg_final_pr:.4f}")
    print(f"Average Accuracy: {avg_final_acc:.4f}")
    print(f"Average F1-score: {avg_final_f1:.4f}")
    print(f"Average Best ROC (within epochs): {avg_best_roc_in_fold:.4f}")
else:
    print("No fold results to average.")

print("\nSimplified model training and evaluation finished.")

# --- 프론트용 d, p ---
# print("\nSaving Drug, Protein Representation...")
# han_model_for_front = HAN_DTI(
#     all_meta_paths=all_meta_paths,
#     in_size=[hd_in_size, hp_in_size],
#     hidden_size=[han_hidden_size, han_hidden_size],
#     out_size=[han_out_size, han_out_size],
#     dropout=dropout_han
# ).to(args['device'])
# han_model_for_front.eval()
# with torch.no_grad():
#     d, p = han_model_for_front(graph_list, initial_node_features[0], initial_node_features[1])
# repr_dir = "../representaion_dp"
# os.makedirs(repr_dir, exist_ok=True)
# torch.save(d, os.path.join(repr_dir, f"{dataset_name}_drug_repr.pt"))
# torch.save(p, os.path.join(repr_dir, f"{dataset_name}_protein_repr.pt"))
# print("\nRepresentation finished.")