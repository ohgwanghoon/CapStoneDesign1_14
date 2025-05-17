import matplotlib.pyplot as plt
import pickle
import os
from sklearn.metrics import roc_curve, precision_recall_curve, auc

save_dir = "../modelSave_part_a"
dataset_name = "heter" # 또는 "Es" 등 원하는 데이터셋 이름
history_file_path = os.path.join(save_dir, f"{dataset_name}_all_folds_history.pkl")

loaded_history = None
with open(history_file_path, 'rb') as f:
    loaded_history = pickle.load(f)

if loaded_history and "fold_1" in loaded_history:
    fold_data = loaded_history["fold_3"]
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(fold_data['epochs'], fold_data['train_loss'], label='Train Loss')
    plt.plot(fold_data['epochs'], fold_data['test_loss'], label='Test Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Fold 1: Loss Curve')

    # 예시: 첫 번째 fold의 ROC/PR 커브를 위한 데이터 (마지막 평가 시점)
    test_labels_f1 = fold_data['test_true_labels'][-1] # 마지막 평가 시점의 레이블
    test_scores_f1 = fold_data['test_pred_scores'][-1] # 마지막 평가 시점의 점수

    fpr, tpr, _ = roc_curve(test_labels_f1, test_scores_f1)
    roc_auc = auc(fpr, tpr)
    plt.subplot(1, 2, 2)
    plt.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Fold 1: ROC Curve (Last Eval)')
    plt.legend(loc="lower right")
    plt.show()