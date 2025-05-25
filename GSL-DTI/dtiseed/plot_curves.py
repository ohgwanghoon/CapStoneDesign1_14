import matplotlib.pyplot as plt
import pickle
import os
import numpy as np
from sklearn.metrics import f1_score

# --- 1. 설정 ---
save_dir = "../result/eval_history"
dataset_name = "heter"

# 비교할 초기 임베딩 방식 및 파일 정보
embedding_experiments = [
    {"suffix": "random", "label": "Random Init", "color": "blue", "linestyle": "-"},
    {"suffix": "pretrained", "label": "Pretrained Init", "color": "green", "linestyle": "--"},
    {"suffix": "similarity", "label": "Similarity Init", "color": "red", "linestyle": "-."},
    {"suffix": "auto_encoder", "label": "Autoencoder Init", "color": "purple", "linestyle": ":"}
]
# 각 실험별 fold_1 데이터를 저장할 딕셔너리
all_experiments_fold1_data = {}

# --- 2. 각 실험 결과(.pkl 파일) 로드 및 fold_1 데이터 추출 ---
for exp_info in embedding_experiments:
    history_file_path = os.path.join(save_dir, f"{exp_info['suffix']}/{dataset_name}_best_fold_history_{exp_info['suffix']}.pkl")
    loaded_history = None
    try:
        with open(history_file_path, 'rb') as f:
            loaded_history = pickle.load(f)
        print(f"Successfully loaded history from: {history_file_path}")

        if loaded_history:
            fold_data = loaded_history
            # F1을 매 에폭마다 재계산 (저장된 true_labels와 pred_scores 사용)
            # fold_data에 'test_f1' 키가 없다면 새로 생성
            if 'test_f1' not in fold_data:
                fold_data['test_f1'] = []
                if fold_data.get('test_true_labels') and fold_data.get('test_pred_scores'):
                    for true_labels, pred_scores in zip(fold_data['test_true_labels'], fold_data['test_pred_scores']):
                        # 확률 점수를 이진 클래스로 변환 (임계값 0.5 사용)
                        threshold = 0.5
                        pred_classes = (np.array(pred_scores) >= threshold).astype(int)
                        f1 = f1_score(true_labels, pred_classes, zero_division=0)
                        fold_data['test_f1'].append(f1)
                else:
                    print(f"Warning: Missing 'test_true_labels' or 'test_pred_scores' for {exp_info['label']} to calculate Acc/F1.")

            all_experiments_fold1_data[exp_info['label']] = fold_data
        else:
            print(f"Warning: 'fold_1' data not found in {history_file_path}")
            all_experiments_fold1_data[exp_info['label']] = None # 데이터 없음을 표시

    except FileNotFoundError:
        print(f"INFO: History file not found (skipping): {history_file_path}")
        all_experiments_fold1_data[exp_info['label']] = None
    except Exception as e:
        print(f"Error loading or processing file {history_file_path}: {e}")
        all_experiments_fold1_data[exp_info['label']] = None

# --- 3. 6개의 그래프 생성 ---
if not any(all_experiments_fold1_data.values()): # 로드된 데이터가 하나도 없으면 종료
    print("No data loaded for any experiment. Exiting plot generation.")
    exit()

fig, axs = plt.subplots(3, 2, figsize=(15, 18)) # 3행 2열의 subplot, 전체 그림 크기 조절
fig.suptitle(f'{dataset_name.upper()} Dataset: Performance Comparison by Initial Embedding', fontsize=18, y=1.00)

plot_metrics = [
    ('train_loss', 'Train Loss'),
    ('test_loss', 'Test Loss'),
    ('test_roc', 'Test AUROC'),
    ('test_pr', 'Test AUPR'),
    ('test_acc', 'Test Accuracy'),
    ('test_f1', 'Test F1-Score')   # 새로 계산된 값 사용
]

# 각 subplot 위치 (axs는 2D 배열이므로 평탄화하여 사용)
ax_flat = axs.flatten()

for idx, (metric_key, metric_label) in enumerate(plot_metrics):
    current_ax = ax_flat[idx]
    for exp_info in embedding_experiments:
        exp_label = exp_info['label']
        fold_data = all_experiments_fold1_data.get(exp_label)

        if fold_data and fold_data.get('epochs') and fold_data.get(metric_key) and len(fold_data['epochs']) == len(fold_data[metric_key]):
            current_ax.plot(fold_data['epochs'], fold_data[metric_key],
                            label=exp_label, color=exp_info['color'], linestyle=exp_info['linestyle'],
                            marker='.', markersize=0.5, alpha=0.8)
        elif fold_data: # 데이터는 로드되었으나 해당 메트릭이 없거나 길이가 안맞는 경우
             print(f"Warning: Data for metric '{metric_key}' incomplete or missing for '{exp_label}'. Skipping plot for this metric.")


    # current_ax.set_xlabel('Epochs')
    current_ax.set_ylabel(metric_label)
    current_ax.set_title(f'{metric_label} Curve')
    current_ax.legend(fontsize='small')
    current_ax.grid(True)
    # current_ax.set_ylim([0.0, 1.0])
    # current_ax.set_xlim([0, 1000])

# 사용되지 않은 subplot이 있다면 숨기기 (만약 plot_metrics 개수보다 subplot이 많다면)
for i in range(len(plot_metrics), len(ax_flat)):
    fig.delaxes(ax_flat[i])

plt.show()