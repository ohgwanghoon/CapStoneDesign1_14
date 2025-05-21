#변경사항
# 1. pretrained embedding의 feature dimension을 모델에 넣을때 MLP를 이용해 조정
# 2. han 결과의 representaion을 이용한 inference를 하도록 함.

import torch
import torch.nn as nn
import torch.nn.functional as F
from modeltestdtiseed import HAN_DTI, MLP, init
import sys

#모델 임베딩 차원 조정(MLP 사용)
class FeatureAlignMLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dims=None):
        super(FeatureAlignMLP, self).__init__()
        
        if hidden_dims is None:
            hidden_dims = [max(input_dim, output_dim)]  # 기본: 입력 또는 출력 중 더 큰 차원 하나의 은닉층

        layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.ReLU())
            prev_dim = h_dim
        
        layers.append(nn.Linear(prev_dim, output_dim))  # 최종 출력 차원 맞추기

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

class DTI_Model(nn.Module):
    def __init__(self, all_meta_paths, initial_feature_in_size, han_hidden_size, han_out_size, mlp_input_dim, dropout_han, init_dim_change = False, raw_drug_feature_dim = 1, raw_protein_feature_dim = 1): 
        # init_dim_change = random feature과 다른feature를 쓸때 TURE, raw_drug_feature_dim, raw_protein_feature_dim 조정)
        """
        Part a로 부터 얻은 노드 임베딩을 이용하여 DTI 예측을하는 MLP가 포함된 DTI 모델
        Args:
            all_meta_paths: Meta-paths for HAN_DTI.
            initial_feature_in_size: List of input feature dimensions for drug/protein initial features [drug_feat_dim, prot_feat_dim].
            han_hidden_size: List of hidden dimensions for drug/protein HAN modules [d_hid, p_hid].
            han_out_size: List of output dimensions from drug/protein HAN modules [d_out, p_out].
                               This will be [drug_embedding_dim, protein_embedding_dim].
            mlp_input_dim: Dimension of the concatenated (drug_embedding + protein_embedding) vector fed to MLP.
                           Should be han_out_size[0] + han_out_size[1].
            dropout_han: Dropout for HAN_DTI module.
            init_dim_change : random feature과 다른 feature를 쓸때 TURE
            raw_drug_feature_dim: random feature과 다른 feature를 쓸때 조정될 drug feature의 차원 수
            raw_protein_feature_dim: random feature과 다른 feature를 쓸때 조정될 protein feature의 차원 수
        """
        super(DTI_Model, self).__init__()

        # Part a: HAN_DTI를 이용하여 drug, protein 임베딩 계산
        self.han_dti = HAN_DTI(
            all_meta_paths=all_meta_paths,
            in_size=initial_feature_in_size,
            hidden_size=han_hidden_size,
            out_size=han_out_size,
            dropout=dropout_han
        )
        self.mlp = MLP(nfeat=mlp_input_dim) # 마지막에 logSoftMax까지 한 결과 나옴
        self.featureAlignMLP_drug = FeatureAlignMLP(raw_drug_feature_dim, initial_feature_in_size[0])
        self.featureAlignMLP_protein = FeatureAlignMLP(raw_protein_feature_dim, initial_feature_in_size[1])
        self.init_dim_change = init_dim_change

    def forward(self, graph_list, initial_node_features, mode, dti_pairs_indices = None, drug_repr = None, protein_repr = None, infer_idx=None):
        """
        DTI 예측을 위한 Forward pass

        Args:
        graph_list: List of DGL graphs [dg, pg].
        initial_node_features: List of initial node feature tensors [drug_features, protein_features].
        mode : train / test / infer . choose one of them.
        dti_pairs_indices: Tensor of shape (N, 2) where N is the number of DTI pairs to predict.
                               Each row is [drug_index, protein_index].
        drug_repr: final representation of drug
        protein_repr: final representation of protein
        infer_idx = index of drugs and proteins which will be used in inference. N * [drug_index, protein_index]
                               
        Returns:
            Output from MLP (log-softmax probabilities for interaction).
        """

        # train or test
        if mode == 'train' or mode == 'test':
            if mode == 'train':
                # 0. random feature과 다른 feature를 쓸때 feature 차원 조정
                if self.init_dim_change == True:
                    drug_input = self.featureAlignMLP_drug(initial_node_features[0])
                    protein_input = self.featureAlignMLP_protein(initial_node_features[1])
                else:
                    drug_input = initial_node_features[0]
                    protein_input = initial_node_features[1]

                # 1. HAN_DTI를 이용하여 drug, protein의 임베딩 계산
                # drug_repr shape : [num_drugs, drug_embedding_dim]
                # protein_repr shape : [num_proteins, protein_embedding_dim]
                drug_repr, protein_repr = self.han_dti(graph_list, drug_input, protein_input)

            # 2. 주어진 DTI pair에 대해 feature 준비
            # dti_pairs_indices의 각 행에서 drug 인덱스와 protein 인덱스를 가져옴
            drug_indices_for_pairs = dti_pairs_indices[:, 0]
            protein_indices_for_pairs = dti_pairs_indices[:, 1]

            # 해당 인덱스를 사용하여 각 pair에 대한 drug 임베딩, protein 임베딩을 선택
            selected_drug_embeddings = drug_repr[drug_indices_for_pairs]
            selected_protein_embeddings = protein_repr[protein_indices_for_pairs]

            # 3. drug 임베딩과 protein 임베딩을 concat
            # pair_features shape : [N, drug_embedding_dim + protein_embedding_dim]
            pair_features = torch.cat((selected_drug_embeddings, selected_protein_embeddings), dim=1)

            # 4. 예측을 위해 pair_features를 MLP
            output = self.mlp(pair_features)

        elif mode == 'infer':
            # infer_idx의 각 행에서 drug 인덱스와 protein 인덱스를 가져옴
            drug_indices_for_pairs = infer_idx[:, 0]
            protein_indices_for_pairs = infer_idx[:, 1]

            # 해당 인덱스를 사용하여 각 pair에 대한 drug 임베딩, protein 임베딩을 선택
            selected_drug_embeddings = drug_repr[drug_indices_for_pairs]
            selected_protein_embeddings = protein_repr[protein_indices_for_pairs]

            # 3. drug 임베딩과 protein 임베딩을 concat
            # pair_features shape : [N, drug_embedding_dim + protein_embedding_dim]
            pair_features = torch.cat((selected_drug_embeddings, selected_protein_embeddings), dim=1)

            # 4. 예측을 위해 pair_features를 MLP
            output = self.mlp(pair_features)

        else:
            sys.exit("Invalid mode. Terminating program.")

        if mode == 'train':
            return output, drug_repr, protein_repr
        else:
            return output