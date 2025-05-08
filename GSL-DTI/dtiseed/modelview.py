import torch

model_path = "../modelSave/heter_fold0_best_roc_model.pt"
device = "cpu" # 단순 확인을 위해 cpu 사용

print(f"Loading model state_dict from: {model_path}")

try:
    state_dict = torch.load(model_path, map_location=device) # cpu에 로드
    print(f"\nLoaded data type: {type(state_dict)}") # 타입 확인
    print("\n--- Parameter Name (Keys) ---") # 파라미터들의 이름 출력 (딕셔너리 키)
    for key in state_dict.keys():
        print(key)

    print("\n-- Parameter Shapes (Top 10) ---") # 파라미터의 이름과 크기 출력 (상위 10개)
    count = 0
    for key, value in state_dict.items():
        if isinstance(value, torch.Tensor):
            print(f"{key}: \tshape={value.shape}, \tdtype={value.dtype}")
        else:
            print(f"{key}: \t(Not a Tensor, type={type(value)})")
        count = count + 1
        if count >= 10:
            break

    # 특정 파라미터의 실제 값 확인
    specific_key = 'MLP.MLP.0.weight' # 보고 싶은 파라미터 이름
    if specific_key in state_dict:
        print(f"\n--- Values for {specific_key} ---")
        print(state_dict[specific_key])

except FileNotFoundError:
    print(f"Error: 파일을 찾을 수 없습니다 - {model_path}")
except Exception as e:
    print(f"Error loading file: {e}")