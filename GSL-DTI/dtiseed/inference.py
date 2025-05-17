import torch
import numpy as np
from utilsdtiseed import load_dataset, setup, default_configure
from modeltestdtiseed import HAN_DTI, MLP, init
from modeltestdtiseed_part_a import DTI_Model
import os

# --- 1. 설정 ---
MODEL_FILE_PATH = "../modelSave_part_a/heter_fold1_best_roc_part_a.pt" # 모델 저장 경로
DATASET_NAME = "heter"
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

# 모델 학습 시 사용했던 하이퍼파라미터 (정확히 일치해야 함)
IN_SIZE_INITIAL = 512
HAN_HIDDEN_SIZE = 256
HAN_OUT_SIZE = 128
DROPOUT_HAN = 0.2 # 학습 시 설정한 dropout 값 (추론 시에는 model.eval()로 비활성화됨)
MLP_INPUT_DIM_VAL = HAN_OUT_SIZE * 2 # 드러그 임베딩 + 단백질 임베딩 차원

# --- 2. 모델 및 원본 데이터 로드 ---
print(f"Setting up arguments and device: {DEVICE}...")
args = setup(default_configure, seed=47)
args['device'] = DEVICE

print(f"Loading dataset '{DATASET_NAME}' for graph structure and node counts...")
# --- dtidata_original은 실제 레이블 값을 포함하지만, 인퍼런스 시에는 주로 drug/protein 인덱스 참조용
# 또는 인덱스 범위를 확인하는 용도로 사용
_, graph_list_original, num_nodes_original, all_meta_paths_original = load_dataset(DATASET_NAME)

print("Preparing initial node features...")
# 학습 시와 동일한 방식으로 초기 노드 feature 생성 또는 로드
hd_init = torch.randn((num_nodes_original[0], IN_SIZE_INITIAL))
hp_init = torch.randn((num_nodes_original[1], IN_SIZE_INITIAL))
initial_node_features_original = [hd_init.to(DEVICE), hp_init.to(DEVICE)]

if DEVICE != "cpu":
    print("Moving original graphs to device...")
    for i in range(len(graph_list_original)):
        graph_list_original[i] = graph_list_original[i].to(DEVICE)

print(f"Loading trained model from: {MODEL_FILE_PATH}...")
# 모델 아키텍처 정의 (학습 시와 동일한 파라미터로)
model = DTI_Model(
    all_meta_paths=all_meta_paths_original,
    initial_feature_in_size=[IN_SIZE_INITIAL, IN_SIZE_INITIAL],
    han_hidden_size=[HAN_HIDDEN_SIZE, HAN_HIDDEN_SIZE],
    han_out_size=[HAN_OUT_SIZE,HAN_OUT_SIZE],
    mlp_input_dim=MLP_INPUT_DIM_VAL,
    dropout_han=DROPOUT_HAN 
).to(DEVICE)

# 저장된 가중치 로드
try:
    model.load_state_dict(torch.load(MODEL_FILE_PATH, map_location=DEVICE))
except FileNotFoundError:
    print(f"ERROR: Model file not found at {MODEL_FILE_PATH}")
    exit()
except Exception as e:
    print(f"ERROR: Could not load model weights: {e}")
    exit()

model.eval()
print("Model loaded sucessfully and set to evaluation mode.")


# --- 3. 예측한 새로운 drug-protein 쌍 정의 ---
# 인덱스는 학습 데이터 생성 시 사용된 인덱스 체계와 동일해야 함
new_dti_pairs_to_predict = [
    (10, 20),
    (15, 30),
    (0, 106),
    (0, 105),
    (num_nodes_original[0] - 1, num_nodes_original[1] - 1)
]
# 입력을 drug, protein의 인덱스로 받을지 이름으로 받을지에 따라 코드 수정 요망

# 예측할 쌍들을 PyTorch 텐서로 변환
dti_pairs_tensor = torch.tensor(new_dti_pairs_to_predict, dtype=torch.long).to(DEVICE)

print(f"\nPredicting for {dti_pairs_tensor.shape[0]} new DTI pairs...")
with torch.no_grad():
    # 모델의 forward 메소드 호출
    # input: 원본 그래프, 원본 초키 노드 feature, 예측한 DTI pair의 인덱스 텐서
    log_softmax_outputs = model(graph_list_original, initial_node_features_original, dti_pairs_tensor)

    # LogSoftmax 출력을 확률로 변환
    interaction_probabilities = torch.exp(log_softmax_outputs[:, 1])

    # threshold 기준 이진 예측
    threshold = 0.5
    predictions_binary = (interaction_probabilities >= threshold).int()

# --- 4. Top K DTI Score 예측 ---
print("\nCalculating DTI scores for all possible pairs to find Top 10...")

# 모든 가능한 드러그-단백질 쌍 목록 생성
all_possible_pairs_list = []
for i in range(num_nodes_original[0]): # 모든 드러그 인덱스
    for j in range(num_nodes_original[1]): # 모든 단백질 인덱스
        all_possible_pairs_list.append((i, j))

all_possible_pairs_tensor = torch.tensor(all_possible_pairs_list, dtype=torch.long).to(DEVICE)

# 모든 가능한 쌍에 대한 예측 점수를 저장할 리스트
all_pairs_scores_list = []

# 메모리 문제를 피하기 위해 모든 쌍을 배치로 나누어 처리 (선택적이지만 권장)
batch_size_inference = 1024 # 인퍼런스 시 배치 크기 (시스템 메모리에 맞게 조절)
with torch.no_grad():
    for k in range(0, all_possible_pairs_tensor.shape[0], batch_size_inference):
        batch_pairs = all_possible_pairs_tensor[k : k + batch_size_inference]
        
        # 모델 forward 호출 (Simplified_DTI_Model 사용)
        log_softmax_outputs_batch = model(graph_list_original, initial_node_features_original, batch_pairs)
        interaction_probabilities_batch = torch.exp(log_softmax_outputs_batch[:, 1]) # 클래스 1 (상호작용) 확률
        
        all_pairs_scores_list.extend(zip(batch_pairs.tolist(), interaction_probabilities_batch.tolist()))
        if (k // batch_size_inference) % 10 == 0 : # 진행 상황 표시 (선택적)
             print(f"  Processed {k + batch_pairs.shape[0]}/{all_possible_pairs_tensor.shape[0]} pairs...")

# 점수가 높은 순으로 정렬
all_pairs_scores_list.sort(key=lambda x: x[1], reverse=True)

# 상위 10개 선택
top_10_dti = all_pairs_scores_list[:10]

# --- 5. 결과 출력 ---
print("\n--- Prediction Results ---")
for i in range(dti_pairs_tensor.shape[0]):
    drug_idx = new_dti_pairs_to_predict[i][0]
    protein_idx = new_dti_pairs_to_predict[i][1]
    score = interaction_probabilities[i].item()
    binary_pred = predictions_binary[i].item()

    print(f"Pair {drug_idx}, Protein {protein_idx}: \tScore = {score:.4f} \tPredicted Interaction = {'Yes' if binary_pred == 1 else 'No'}")

output_dir = "../inference_result"
os.makedirs(output_dir, exist_ok=True)
results_file_path = os.path.join(output_dir, f"dti_prediction.txt")
top10_file_path = os.path.join(output_dir, f"top10_dti_prediction.txt")
print(f"\n--- Predicition Results (saving to {results_file_path})")

try:
    with open(results_file_path, 'w', encoding='utf-8') as f_out:
        f_out.write("Drug_Index\tProtein_Index\tInteraction_Score\tPredicted_Interaction\n")

        for i in range(dti_pairs_tensor.shape[0]): # dti_pairs_tensor는 예측할 쌍들의 텐서
            drug_idx = new_dti_pairs_to_predict[i][0] # new_dti_pairs_to_predict는 파이썬 리스트
            protein_idx = new_dti_pairs_to_predict[i][1]
            score = interaction_probabilities[i].item() # 해당 쌍의 상호작용 확률 점수
            
            # 이진 예측 (임계값 기준)
            threshold = 0.5 
            binary_pred_val = 1 if score >= threshold else 0
            binary_pred_label = 'Yes' if binary_pred_val == 1 else 'No'
            
            # 파일에 저장 (탭으로 구분된 형식)
            f_out.write(f"{drug_idx}\t{protein_idx}\t{score:.4f}\t{binary_pred_label}\n")
    
    print(f"\nResults successfully saved to {results_file_path}")

except Exception as e:
    print(f"Error writing results to file: {e}")

try:
    with open(top10_file_path, 'w', encoding='utf-8') as f_out:
        f_out.write("Rank\tDrug_Index\tProtein_Index\tInteraction_Score\n")
        for rank, ((drug_idx, protein_idx), score) in enumerate(top_10_dti):
            # 파일에 저장
            f_out.write(f"{rank+1}\t{drug_idx}\t{protein_idx}\t{score:.4f}\n")
    print(f"\nTop 10 DTI results successfully saved to {top10_file_path}")
except Exception as e:
    print(f"Error writing Top 10 DTI results to file: {e}")