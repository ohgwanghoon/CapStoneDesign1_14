import torch
import numpy as np
from utilsdtiseed import load_dataset, setup, default_configure
from modeltestdtiseed import HAN_DTI, MLP, init
from modeltestdtiseed_part_a import DTI_Model

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

    # thresold 기준 이진 예측
    thresold = 0.5
    predictions_binary = (interaction_probabilities >= thresold).int()

# --- 5. 결과 출력 ---
print("\n--- Prediction Results ---")
for i in range(dti_pairs_tensor.shape[0]):
    drug_idx = new_dti_pairs_to_predict[i][0]
    protein_idx = new_dti_pairs_to_predict[i][1]
    score = interaction_probabilities[i].item()
    binary_pred = predictions_binary[i].item()

    print(f"Pair {drug_idx}, Protein {protein_idx}: \tScore = {score:.4f} \tPredicted Interaction = {'Yes' if binary_pred == 1 else 'No'}")