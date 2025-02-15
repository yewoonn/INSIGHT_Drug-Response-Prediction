import torch
import numpy as np
from itertools import product
from sklearn.model_selection import train_test_split

def filter_data(sample_indices, gene_embeddings, drug_graphs, substructure_embeddings, labels_dict):
    """
    sample_indices에 해당하는 (cell_line, drug) 쌍에 대해
    필요한 gene_embeddings, drug_graphs, substructure_embeddings, labels 를 필터링하여 반환합니다.
    """
    filtered_gene_embeddings = {cell_line: gene_embeddings[cell_line] for cell_line, _ in sample_indices}
    filtered_drug_graphs = {drug: drug_graphs[drug] for _, drug in sample_indices}
    filtered_substructure_embeddings = {
        drug: substructure_embeddings[drug] for _, drug in sample_indices
    }
    filtered_labels = {(cell_line, drug): labels_dict[(cell_line, drug)] for cell_line, drug in sample_indices}
    return filtered_gene_embeddings, filtered_drug_graphs, filtered_substructure_embeddings, filtered_labels

# =================================================================================================

# gene_embeddings
gene_embeddings = torch.load('/data1/project/seoeun/data/#312_gene_embeddings.pt')

# drug_embeddings
saved_embeddings = np.load('/data1/project/seoeun/data/drug_embeddings_10.npz')
drug_embeddings = {
    key: torch.tensor(saved_embeddings[key], dtype=torch.int64)
    for key in saved_embeddings.keys()
}
saved_embeddings.close()

# drug_graph_dict
drug_graph_dict = torch.load('/data1/project/seoeun/data/drug_graph_dict_10.pt')

# labels_dict
labels_dict = torch.load('/data1/project/seoeun/data/drug_label_dict_10.pt')

# 실제 cell line, drug 목록
cell_lines = list(gene_embeddings.keys())
drugs = list(drug_embeddings.keys())
print("cell_lines : ", len(cell_lines), "drugs : ", len(drugs))

# 데이터셋 분할 비율 (drug 기준)
train_ratio = 0.8
val_ratio = 0.1
test_ratio = 0.1  # (1 - train_ratio - val_ratio)

# drug 분할
train_drugs, temp_drugs = train_test_split(
    drugs, test_size=(1 - train_ratio), random_state=42
)

val_drugs, test_drugs = train_test_split(
    temp_drugs,
    test_size=(test_ratio / (val_ratio + test_ratio)),  # 또는 1 - (val_ratio / (val_ratio + test_ratio))
    random_state=42
)

# 모든 cell line과 해당 drug들의 데카르트 곱으로 sample_indices 구성
train_indices = list(product(cell_lines, train_drugs))
val_indices = list(product(cell_lines, val_drugs))
test_indices = list(product(cell_lines, test_drugs))

print(f"Train set drugs: {len(train_drugs)} -> train_samples: {len(train_indices)}")
print(f"Validation set drugs: {len(val_drugs)} -> val_samples: {len(val_indices)}")
print(f"Test set drugs: {len(test_drugs)} -> test_samples: {len(test_indices)}")

# 필터링
train_gene_embeddings, train_drug_graphs, train_drug_embeddings, train_labels = filter_data(
    train_indices, gene_embeddings, drug_graph_dict, drug_embeddings, labels_dict
)
val_gene_embeddings, val_drug_graphs, val_drug_embeddings, val_labels = filter_data(
    val_indices, gene_embeddings, drug_graph_dict, drug_embeddings, labels_dict
)
test_gene_embeddings, test_drug_graphs, test_drug_embeddings, test_labels = filter_data(
    test_indices, gene_embeddings, drug_graph_dict, drug_embeddings, labels_dict
)

# .pt 파일로 저장
torch.save({
    'gene_embeddings': train_gene_embeddings,
    'drug_embeddings': train_drug_embeddings,
    'drug_graphs': train_drug_graphs,
    'labels': train_labels,
    'sample_indices': train_indices,
}, '../dataset/train_dataset.pt')

torch.save({
    'gene_embeddings': val_gene_embeddings,
    'drug_embeddings': val_drug_embeddings,
    'drug_graphs': val_drug_graphs,
    'labels': val_labels,
    'sample_indices': val_indices,
}, '../dataset/val_dataset.pt')

torch.save({
    'gene_embeddings': test_gene_embeddings,
    'drug_embeddings': test_drug_embeddings,
    'drug_graphs': test_drug_graphs,
    'labels': test_labels,
    'sample_indices': test_indices,
}, '../dataset/test_dataset.pt')


print("Unseen Drug Split : Datasets saved to .pt files.")
