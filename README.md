# CapStoneDesign1_14

## 프로젝트 개요

이 프로젝트는 다양한 초기 임베딩 전략을 실험하여 약물-타겟 상호작용(DTI) 예측을 위한 단백질 및 약물 표현을 개선하는 것을 목표로 합니다. 다양한 초기화 방법이 DTI 모델의 성능에 미치는 영향을 평가합니다.

핵심 구현 및 데이터 리소스는 [GSL-DTI](./GSL-DTI/) 디렉토리에 정리되어 있습니다.

## 임베딩 전략

본 프로젝트에서는 약물과 단백질에 대해 다음과 같은 초기 임베딩 방법을 실험합니다:

### 1. 유사도 기반 임베딩
- 약물 간 화학적 구조 유사도, 단백질 간 1차 구조 유사도를 feature로 사용합니다.
- 자세한 내용은 [similarity_introduction.txt](./GSL-DTI/discription_front/similarity_introduction.txt) 파일을 참고하세요.

### 2. 오토인코더 기반 임베딩
- 오토인코더를 통해 추출한 약물/단백질 유사도 행렬의 잠재 공간(latent space)을 초기 feature로 사용합니다.
- 약물과 단백질 유사도 정보의 핵심적인 내용을 feature에 내포합니다.
- 자세한 내용은 [autoEncoder_introduction.txt](./GSL-DTI/discription_front/autoEncoder_introduction.txt) 파일을 참고하세요.

### 3. 사전학습(pretrained) 기반 임베딩
- 약물과 단백질의 물리적 구조를 초기 feature에 반영합니다.
- 약물: SMILES를 chemBERTa에 입력하여 분자적 특징을 가진 representation을 생성합니다.
- 단백질: sequence를 ESM에 입력하여 구조적 특징을 가진 representation을 생성합니다.
- 자세한 내용은 [pretrained_introduction.txt](./GSL-DTI/discription_front/pretrained_introduction.txt) 파일을 참고하세요.

### 4. 랜덤 기반 임베딩
- 표준 정규 분포를 따르는 랜덤 난수를 feature로 사용합니다.
- 자세한 내용은 [random_introduction.txt](./GSL-DTI/discription_front/random_introduction.txt) 파일을 참고하세요.

## 디렉토리 구조

- **GSL-DTI/**: 모든 임베딩 전략의 코드, 데이터, 문서가 포함된 메인 디렉토리입니다.
  - **discription_front/**: 각 임베딩 방법에 대한 상세 소개가 포함되어 있습니다.
  - **data/**: 약물 및 단백질의 데이터셋과 유사도 행렬이 포함되어 있습니다.
  - **dtiseed/**: 모델 학습, 평가, 임베딩 생성을 위한 소스 코드가 포함되어 있습니다.
  - **init_feature/**: 약물 및 단백질의 사전 계산된 임베딩 feature가 포함되어 있습니다.
  - **result/**: 실험 결과 및 모델 출력이 포함되어 있습니다.

각 임베딩 전략 및 구현 세부사항에 대한 추가 정보는 [GSL-DTI/discription_front](./GSL-DTI/discription_front/) 디렉토리의 해당 파일을 참고하세요.

## 실험 설정 및 실행 방법

### 1. 실험 환경 및 주요 스크립트
- 실험은 `GSL-DTI/dtiseed/main.py`에서 수행합니다.
- 주요 하이퍼파라미터(임베딩 차원, hidden size, learning rate, epoch 등)는 main.py 상단에서 설정할 수 있습니다.
- 실험에 사용할 데이터셋(`dataset_name`)과 임베딩 방식(`init_feature_flag`)을 main.py에서 선택할 수 있습니다.

### 2. 임베딩 방식 선택
- `init_feature_flag` 값에 따라 초기 임베딩이 결정됩니다:
    - 0: 랜덤 임베딩 (`init_feature/drug_embedding_random.pt`, `protein_embedding_random.pt`)
    - 1: 사전학습 임베딩 (`init_feature/drug_chemBERTa_embeddings.pt`, `protein_esm_embeddings.pt`)
    - 2: 유사도 기반 임베딩 (`data/heter/Similarity_Matrix_Drugs.txt`, `Similarity_Matrix_Proteins.txt`)
    - 3: 오토인코더 임베딩 (`init_feature/drug_embedding_autoencoder.pt`, `protein_embedding_autoencoder.pt`)

### 3. 데이터 및 임베딩 파일
- 입력 데이터는 `GSL-DTI/data/` 하위 폴더에 저장되어 있습니다.
- 임베딩 파일은 `GSL-DTI/init_feature/`에 위치합니다.
- 실험 결과 및 모델 가중치는 `GSL-DTI/result/`에 저장됩니다.

### 4. 학습 및 평가
- 5-fold 교차 검증(StratifiedKFold)으로 실험을 수행합니다.
- 각 fold별로 모델이 초기화되어 학습 및 평가가 진행되며, 최적 성능(최저 loss) 모델 가중치가 저장됩니다.
- 조기 종료(Early Stopping, patience=10)가 적용되어, 성능 개선이 없을 경우 학습이 조기에 종료됩니다.
- 각 fold의 결과(ROC, PR, F1, Accuracy 등)는 `result/eval_history/`에 pickle 파일로 저장됩니다.
- 학습이 완료되면, 각 fold별로 약물/단백질 임베딩 결과(`result/drug_protein_embed/`)도 저장됩니다.

### 5. 실험 실행 예시
```bash
cd GSL-DTI/dtiseed
python main.py
```
- 필요에 따라 main.py 내 하이퍼파라미터, 데이터셋, 임베딩 방식을 수정한 후 실행하세요.

## 인용 안내

본 프로젝트 또는 GSL-DTI를 연구에 활용하실 경우, 적절히 인용해 주시기 바랍니다.
