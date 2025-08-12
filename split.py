import torch
from itertools import product
from sklearn.model_selection import train_test_split

# 필터링 함수
def filter_data(sample_indices, gene_embeddings, drug_embeddings, drug_graphs, drug_masks, labels_dict):
    filtered_gene_embeddings = {cell_line: gene_embeddings[cell_line] for cell_line, _ in sample_indices}
    filtered_drug_embeddings = {drug: drug_embeddings[drug] for _, drug in sample_indices}
    filtered_drug_graphs = {drug: drug_graphs[drug] for _, drug in sample_indices}
    filtered_drug_masks = {drug: drug_masks[drug] for _, drug in sample_indices}
    filtered_labels = {(cell_line, drug): labels_dict[(cell_line, drug)] for cell_line, drug in sample_indices}
    return filtered_gene_embeddings, filtered_drug_embeddings, filtered_drug_graphs, filtered_drug_masks, filtered_labels

# 데이터 로드
gene_embeddings = torch.load('./input/gene_embeddings_10_fold_binary.pt') 
drug_embeddings = torch.load('./input/drug_embeddings.pt')
drug_substructure_embeddings = torch.load('./input/drug_BRICS_embeddings.pt')
drug_substructure_masks = torch.load('./input/drug_BRICS_masks.pt')
labels_dict = torch.load('./input/response_label_dict_LN.pt')

cell_lines = list(gene_embeddings.keys())
drugs = list(drug_embeddings.keys())
sample_indices = list(labels_dict.keys())

# 데이터셋 분할 비율 설정 (3:1:1)
train_ratio = 0.6
val_ratio = 0.2
test_ratio = 0.2

# Train, Validation, Test 분할
train_indices, temp_indices = train_test_split(
    sample_indices, test_size=(1 - train_ratio), random_state=42
)
val_indices, test_indices = train_test_split(
    temp_indices, test_size=test_ratio / (val_ratio + test_ratio), random_state=42
)


test_gene_emb, test_drug_emb, test_drug_graphs, test_masks, test_labels = filter_data(
    test_indices, gene_embeddings, drug_embeddings, drug_substructure_embeddings, drug_substructure_masks, labels_dict
)

# Test 데이터 별도 저장
test_dataset = {
    'gene_embeddings': test_gene_emb,
    'drug_embeddings': test_drug_emb,
    'drug_substructure_embeddings': test_drug_graphs,
    'drug_substructure_masks': test_masks,
    'labels': test_labels,
    'sample_indices': test_indices,
}
torch.save(test_dataset, './dataset/test_dataset.pt')

# Fold 인덱스 생성 및 저장 (5-fold cross-validation)
from sklearn.model_selection import KFold

kf = KFold(n_splits=5, shuffle=True, random_state=42)
train_val_indices = train_indices + val_indices

for fold_num, (train_idx, val_idx) in enumerate(kf.split(train_val_indices)):
    current_train_indices = [train_val_indices[i] for i in train_idx]
    current_val_indices = [train_val_indices[i] for i in val_idx]

    # 폴드별 데이터 저장
    fold_train_gene_emb, fold_train_drug_emb, fold_train_drug_graphs, fold_train_masks, fold_train_labels = filter_data(
        current_train_indices, gene_embeddings, drug_embeddings, drug_substructure_embeddings, drug_substructure_masks, labels_dict
    )

    fold_val_gene_emb, fold_val_drug_emb, fold_val_drug_graphs, fold_val_masks, fold_val_labels = filter_data(
        current_val_indices, gene_embeddings, drug_embeddings, drug_substructure_embeddings, drug_substructure_masks, labels_dict
    )

    fold_dataset = {
        'train': {
            'gene_embeddings': fold_train_gene_emb,
            'drug_embeddings': fold_train_drug_emb,
            'drug_substructure_embeddings': fold_train_drug_graphs,
            'drug_substructure_masks': fold_train_masks,
            'labels': fold_train_labels,
            'sample_indices': current_train_indices,
        },
        'validation': {
            'gene_embeddings': fold_val_gene_emb,
            'drug_embeddings': fold_val_drug_emb,
            'drug_substructure_embeddings': fold_val_drug_graphs,
            'drug_substructure_masks': fold_val_masks,
            'labels': fold_val_labels,
            'sample_indices': current_val_indices,
        }
    }

    torch.save(fold_dataset, f'./dataset/cross_valid_fold_{fold_num + 1}.pt')

print("전체 데이터셋 및 각 Fold 데이터셋과 Test 데이터셋이 저장되었습니다.")
