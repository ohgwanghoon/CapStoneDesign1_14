import torch
import numpy as np
import os

def load_data(filename):
    ext = os.path.splitext(filename)[1]
    if ext == '.pt':
        data = torch.load(filename)
        if not isinstance(data, torch.Tensor):
            raise TypeError("'.pt' 파일은 torch.Tensor 형식이어야 합니다.")
        return data.numpy()
    elif ext == '.txt':
        return np.loadtxt(filename)
    else:
        raise ValueError("지원하지 않는 파일 형식입니다. '.pt' 또는 '.txt'만 가능합니다.")

def standardize(data):
    mean = data.mean()
    std = data.std()
    return (data - mean) / (std + 1e-8)

def save_data(data, output_filename):
    np.savetxt(output_filename, data, fmt='%.6f')
    print(f"표준 정규화된 데이터를 '{output_filename}'로 저장하였습니다.")

def main(input_filename, output_filename):
    data = load_data(input_filename)
    standardized = standardize(data)
    save_data(standardized, output_filename)

# 사용 예시
if __name__ == '__main__':
    input_file = '../data/heter/Similarity_Matrix_Proteins.txt'   
    output_file = '../init_feature/standard_protein_similarity.txt'
    main(input_file, output_file)
