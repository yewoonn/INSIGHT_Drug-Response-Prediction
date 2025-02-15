import torch
import numpy as np
from itertools import product
from sklearn.model_selection import train_test_split

# 필터링 함수 (원본과 동일)
def filter_data(sample_indices, gene_embeddings, drug_graphs, substructure_embeddings, labels_dict):
    filtered_gene_embeddings = {cell_line: gene_embeddings[cell_line] for cell_line, _ in sample_indices}
    filtered_drug_graphs = {drug: drug_graphs[drug] for _, drug in sample_indices}
    filtered_substructure_embeddings = {
        drug: substructure_embeddings[drug] for _, drug in sample_indices
    }
    filtered_labels = {(cell_line, drug): labels_dict[(cell_line, drug)] for cell_line, drug in sample_indices}
    return filtered_gene_embeddings, filtered_drug_graphs, filtered_substructure_embeddings, filtered_labels

# =================================================================================================

gene_embeddings = torch.load('/data1/project/seoeun/data/#312_gene_embeddings.pt')

saved_embeddings = np.load('/data1/project/seoeun/data/drug_embeddings_10.npz')
drug_embeddings = {
    key: torch.tensor(saved_embeddings[key], dtype=torch.int64)
    for key in saved_embeddings.keys()
}
saved_embeddings.close()

drug_graph_dict = torch.load('/data1/project/seoeun/data/drug_graph_dict_10.pt')
labels_dict = torch.load('/data1/project/seoeun/data/drug_label_dict_10.pt')

cell_lines = list(gene_embeddings.keys())  # 실제 cell line 목록
drugs = list(drug_embeddings.keys())       # 실제 drug 목록

print("cell_lines : ", len(cell_lines), "drugs : ", len(drugs))

# 데이터셋 분할 비율
train_ratio = 0.8
val_ratio = 0.1
test_ratio = 0.1  # (1 - train_ratio - val_ratio)

# 1. cell_lines를 먼저 train, temp로 분할
train_cell_lines, temp_cell_lines = train_test_split(
    cell_lines, test_size=(1 - train_ratio), random_state=42
)

# 2. temp_cell_lines를 val, test로 분할
val_cell_lines, test_cell_lines = train_test_split(
    temp_cell_lines,
    test_size=(test_ratio / (val_ratio + test_ratio)),  # 또는 1 - (val_ratio / (val_ratio + test_ratio))
    random_state=42
)

# 3. 각 분할(cell lines)과 모든 drugs의 데카르트 곱으로 sample_indices 생성
train_indices = list(product(train_cell_lines, drugs))
val_indices = list(product(val_cell_lines, drugs))
test_indices = list(product(test_cell_lines, drugs))

print("Train set cell lines:", len(train_cell_lines),
      "-> train_samples:", len(train_indices))
print("Validation set cell lines:", len(val_cell_lines),
      "-> val_samples:", len(val_indices))
print("Test set cell lines:", len(test_cell_lines),
      "-> test_samples:", len(test_indices))

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

# Save train data
torch.save({
    'gene_embeddings': train_gene_embeddings,
    'drug_embeddings': train_drug_embeddings,
    'drug_graphs': train_drug_graphs,
    'labels': train_labels,
    'sample_indices': train_indices,
}, '../dataset/train_dataset.pt')

# Save validation data
torch.save({
    'gene_embeddings': val_gene_embeddings,
    'drug_embeddings': val_drug_embeddings,
    'drug_graphs': val_drug_graphs,
    'labels': val_labels,
    'sample_indices': val_indices,
}, '../dataset/val_dataset.pt')

# Save test data
torch.save({
    'gene_embeddings': test_gene_embeddings,
    'drug_embeddings': test_drug_embeddings,
    'drug_graphs': test_drug_graphs,
    'labels': test_labels,
    'sample_indices': test_indices,
}, '../dataset/test_dataset.pt')

print("Unseen Cell Line Split : Datasets saved to .pt files.")
