import torch
import torch.nn as nn
import torch.nn.functional as F
from modeltestdtiseed import HAN_DTI, MLP, init

class DTI_Model(nn.Module):
    def __init__(self, all_meta_paths, initial_feature_in_size, han_hidden_size, han_out_size, mlp_input_dim, dropout_han):
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
        self.mlp = MLP(nfeat=mlp_input_dim)

    def forward(self, graph_list, initial_node_features, dti_pairs_indices):
        """
        DTI 예측을 위한 Forward pass

        Args:
        graph_list: List of DGL graphs [dg, pg].
        initial_node_features: List of initial node feature tensors [drug_features, protein_features].
        dti_pairs_indices: Tensor of shape (N, 2) where N is the number of DTI pairs to predict.
                               Each row is [drug_index, protein_index].

        Returns:
            Output from MLP (log-softmax probabilities for interaction).
        """
        # 1. HAN_DTI를 이용하여 drug, protein의 임베딩 계산
        # drug_repr shape : [num_drugs, drug_embedding_dim]
        # protein_repr shape : [num_proteins, protein_embedding_dim]
        drug_repr, protein_repr = self.han_dti(graph_list, initial_node_features[0], initial_node_features[1])

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

        return output