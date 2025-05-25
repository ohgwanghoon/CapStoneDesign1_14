import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import pickle # 사용하지 않지만, main.py와의 일관성을 위해 남겨둘 수 있음

# 필요한 함수 및 클래스 임포트
from utilsdtiseed import load_dataset, setup, default_configure # init 함수는 DTI_Model에서 직접 호출하지 않으므로 제외 가능
# main.py에 최종 DTI_Model 및 FeatureAlignMLP 정의가 있다고 가정하고 임포트
from modeltestdtiseed_rep_infer import DTI_Model, FeatureAlignMLP
# MLP는 DTI_Model 내부에서 사용되지만, 혹시 직접 로드할 경우를 대비해 modeltestdtiseed_rep_infer에서 가져올 수 있음
# from modeltestdtiseed_rep_infer import MLP

# --- 1. 기본 설정 ---
DATASET_NAME = "heter"
# 어떤 Fold의 모델과 표현 벡터를 사용할지 지정
# 모든 fold에 대해 실행하려면 이 부분을 반복문으로 만들거나 스크립트 인자로 받아야 함
TARGET_FOLD_NUMBER = 5 # 예시: Fold 1의 결과 사용

BASE_RESULT_DIR = "../result" # main.py의 save_dir과 일치
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

# 모델 학습 시 사용했던 공통 하이퍼파라미터 (main.py에서 가져옴)
IN_SIZE_INITIAL = 512 # FeatureAlignMLP의 목표 차원 또는 Random/Autoencoder 특징의 기본 차원
HAN_HIDDEN_SIZE = 256
HAN_OUT_SIZE = 128
DROPOUT_HAN = 0.5
MLP_INPUT_DIM_VAL = HAN_OUT_SIZE * 2

# --- 2. 반복할 초기 임베딩 방식 정의 ---
# main.py의 init_feature_flag 및 경로/이름 규칙과 일치
embedding_configs = [
    {"flag_val": 0, "name": "random", "init_dim_change": False,
     "drug_feat_path_load": "../init_feature/drug_embedding_random.pt", # DTI_Model 생성 시 차원 결정용
     "prot_feat_path_load": "../init_feature/protein_embedding_random.pt"},
    {"flag_val": 1, "name": "pretrained", "init_dim_change": True,
     "drug_feat_path_load": "../init_feature/drug_chemBERTa_embeddings.pt",
     "prot_feat_path_load": "../init_feature/protein_esm_embeddings.pt"},
    {"flag_val": 2, "name": "similarity", "init_dim_change": True,
     "drug_feat_path_load": f'../data/{DATASET_NAME}/Similarity_Matrix_Drugs.txt',
     "prot_feat_path_load": f'../data/{DATASET_NAME}/Similarity_Matrix_Proteins.txt'},
    {"flag_val": 3, "name": "auto_encoder", "init_dim_change": False, # main.py 로직에 따라 False
     "drug_feat_path_load": '../init_feature/drug_embedding_autoencoder.pt',
     "prot_feat_path_load": '../init_feature/protein_embedding_autoencoder.pt'}
]

# --- 3. 공통 데이터 로드 (그래프 구조, 노드 수 등) ---
print(f"Setting up arguments and device: {DEVICE}...")
args = setup(default_configure, seed=47) # main.py와 동일 시드
args['device'] = DEVICE

print(f"Loading dataset '{DATASET_NAME}' for graph structure and node counts...")
# 인퍼런스 시에는 dtidata (상호작용 레이블 포함)는 직접 필요 없고,
# graph_list_original, num_nodes_original, all_meta_paths_original 만 사용
_, graph_list_original, num_nodes_original, all_meta_paths_original = load_dataset(DATASET_NAME)

if DEVICE != "cpu": # 그래프는 HAN_DTI에 필요 없지만, 모델 생성 시 all_meta_paths와 함께 전달되므로 형식상 로드
    print("Moving original graphs to device (for model instantiation)...")
    for i in range(len(graph_list_original)):
        graph_list_original[i] = graph_list_original[i].to(DEVICE)
    print("Graphs moved (though not directly used in this inference mode).")


# --- 4. 각 초기 임베딩 방식에 대해 인퍼런스 반복 ---
for config in embedding_configs:
    current_init_flag = config["flag_val"]
    current_model_suffix = config["name"]
    current_init_dim_change = config["init_dim_change"]

    print(f"\n\n--- Processing Inference for Model with '{current_model_suffix}' Embeddings (Fold {TARGET_FOLD_NUMBER}) ---")

    # 4.1. 모델 파일 및 학습된 Representation 파일 경로 설정
    model_weights_path = os.path.join(BASE_RESULT_DIR, f"trained_weight/{current_model_suffix}/{DATASET_NAME}_fold{TARGET_FOLD_NUMBER}_best_loss_modelWeight.pt")
    
    learned_repr_dir = os.path.join(BASE_RESULT_DIR, f"drug_protein_embed/{current_model_suffix}")
    drug_repr_path = os.path.join(learned_repr_dir, f"drug_repr_fold{TARGET_FOLD_NUMBER}.pt")
    protein_repr_path = os.path.join(learned_repr_dir, f"protein_repr_fold{TARGET_FOLD_NUMBER}.pt")

    # 4.2. 결과 저장 폴더 및 파일 경로 설정
    output_inference_subdir = os.path.join(BASE_RESULT_DIR, f"inference_score/{current_model_suffix}")
    os.makedirs(output_inference_subdir, exist_ok=True)
    all_scores_file_path = os.path.join(output_inference_subdir, f"all_pairs_dti_scores_{DATASET_NAME}_fold{TARGET_FOLD_NUMBER}.txt")

    # 4.3. DTI_Model 초기화에 필요한 실제 원본 특징 차원 결정 (main.py 로직과 동일하게)
    # 이 부분은 DTI_Model의 FeatureAlignMLP의 입력 차원을 정확히 설정하기 위함입니다.
    if current_init_flag == 2: # Similarity (NumPy .txt 파일)
        _hd_raw_shape_check = np.loadtxt(config["drug_feat_path_load"])
        _hp_raw_shape_check = np.loadtxt(config["prot_feat_path_load"])
        raw_drug_feature_dim_actual = _hd_raw_shape_check.shape[1]
        raw_protein_feature_dim_actual = _hp_raw_shape_check.shape[1]
    else: # .pt 파일
        _hd_raw_shape_check = torch.load(config["drug_feat_path_load"], map_location='cpu') # 크기 확인용 CPU 로드
        _hp_raw_shape_check = torch.load(config["prot_feat_path_load"], map_location='cpu')
        raw_drug_feature_dim_actual = _hd_raw_shape_check.shape[1]
        raw_protein_feature_dim_actual = _hp_raw_shape_check.shape[1]
    
    model_han_input_drug_dim = IN_SIZE_INITIAL if current_init_dim_change else raw_drug_feature_dim_actual
    model_han_input_protein_dim = IN_SIZE_INITIAL if current_init_dim_change else raw_protein_feature_dim_actual

    # 4.4. DTI_Model 객체 생성 및 학습된 가중치 로드 (MLP 가중치 사용 목적)
    print(f"Loading trained model structure from: {model_weights_path}...")
    # DTI_Model은 HAN_DTI와 MLP를 모두 포함하므로, 전체 구조를 로드해야 state_dict 키가 맞습니다.
    model_for_inference = DTI_Model(
        all_meta_paths=all_meta_paths_original,
        initial_feature_in_size=[model_han_input_drug_dim, model_han_input_protein_dim],
        han_hidden_size=[HAN_HIDDEN_SIZE, HAN_HIDDEN_SIZE],
        han_out_size=[HAN_OUT_SIZE, HAN_OUT_SIZE],
        mlp_input_dim=MLP_INPUT_DIM_VAL,
        dropout_han=DROPOUT_HAN,
        init_dim_change=current_init_dim_change,
        raw_drug_feature_dim=raw_drug_feature_dim_actual,
        raw_protein_feature_dim=raw_protein_feature_dim_actual
    ).to(DEVICE)

    try:
        model_for_inference.load_state_dict(torch.load(model_weights_path, map_location=DEVICE))
    except FileNotFoundError:
        print(f"ERROR: Model weights file not found for '{current_model_suffix}' at {model_weights_path}. Skipping this model.")
        continue
    except Exception as e:
        print(f"ERROR: Could not load model weights for '{current_model_suffix}': {e}. Skipping this model.")
        continue
    
    model_for_inference.eval() # 평가 모드 설정
    print(f"Model for '{current_model_suffix}' loaded.")

    # 4.5. 학습된 드러그 및 단백질 표현 벡터 로드
    print(f"Loading pre-calculated representations for '{current_model_suffix}'...")
    try:
        drug_repr_learned = torch.load(drug_repr_path, map_location=DEVICE)
        protein_repr_learned = torch.load(protein_repr_path, map_location=DEVICE)
        print(f"  Loaded drug embeddings shape: {drug_repr_learned.shape}")
        print(f"  Loaded protein embeddings shape: {protein_repr_learned.shape}")
    except FileNotFoundError:
        print(f"ERROR: Learned representation files not found for '{current_model_suffix}'. Skipping this model.")
        print(f"  Expected Drug Repr: {drug_repr_path}")
        print(f"  Expected Protein Repr: {protein_repr_path}")
        continue
    except Exception as e:
        print(f"Error loading learned representations for '{current_model_suffix}': {e}. Skipping this model.")
        continue

    # 4.6. 모든 가능한 쌍에 대한 DTI Score 예측 및 파일 저장
    print(f"Predicting DTI scores for all pairs using '{current_model_suffix}' representations...")
    all_possible_pairs_list = []
    for i in range(num_nodes_original[0]): # 총 드러그 수
        for j in range(num_nodes_original[1]): # 총 단백질 수
            all_possible_pairs_list.append((i, j))
    all_possible_pairs_tensor = torch.tensor(all_possible_pairs_list, dtype=torch.long).to(DEVICE)

    batch_size_inference = 2048 # 시스템 메모리 상황에 따라 조절

    try:
        with open(all_scores_file_path, 'w', encoding='utf-8') as f_out:
            f_out.write("Drug_Index\tProtein_Index\tInteraction_Score\n") # 헤더

            with torch.no_grad():
                for k_batch in range(0, all_possible_pairs_tensor.shape[0], batch_size_inference):
                    batch_pairs_indices = all_possible_pairs_tensor[k_batch : k_batch + batch_size_inference]
                    
                    # DTI_Model의 forward 메소드 호출 (mode='infer')
                    log_softmax_outputs_batch = model_for_inference(
                        graph_list=None,             # mode='infer'에서는 사용 안 함
                        initial_node_features=None,  # mode='infer'에서는 사용 안 함
                        mode='infer',
                        dti_pairs_indices=None,      # mode='infer'에서는 infer_idx 사용
                        drug_repr=drug_repr_learned, # 로드한 드러그 표현 전달
                        protein_repr=protein_repr_learned, # 로드한 단백질 표현 전달
                        infer_idx=batch_pairs_indices     # 현재 배치에서 예측할 쌍 인덱스
                    )
                    interaction_probabilities_batch = torch.exp(log_softmax_outputs_batch[:, 1])
                    
                    cpu_batch_pairs = batch_pairs_indices.cpu().tolist()
                    cpu_interaction_probabilities = interaction_probabilities_batch.cpu().tolist()
                    
                    for pair_indices_list, score_val in zip(cpu_batch_pairs, cpu_interaction_probabilities):
                        f_out.write(f"{pair_indices_list[0]}\t{pair_indices_list[1]}\t{score_val:.6f}\n")
                    
                    if (k_batch // batch_size_inference) % 50 == 0 : # 50 배치마다 진행 상황 표시
                         print(f"  Processed {k_batch + batch_pairs_indices.shape[0]}/{all_possible_pairs_tensor.shape[0]} pairs for '{current_model_suffix}'...")
            
        print(f"All DTI scores for '{current_model_suffix}' saved to: {all_scores_file_path}")

    except Exception as e:
        print(f"Error during inference or saving scores for '{current_model_suffix}': {e}")

print("\nAll inference processes complete.")