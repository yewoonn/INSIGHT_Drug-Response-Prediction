import torch
import numpy as np
from itertools import product
from sklearn.model_selection import train_test_split

# 필터링 함수 (기존과 동일)
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

cell_lines = list(gene_embeddings.keys())  # 전체 Cell line 목록
drugs = list(drug_embeddings.keys())       # 전체 Drug 목록
print("Total cell_lines:", len(cell_lines), "Total drugs:", len(drugs))

# 분할 비율
cell_line_train_ratio = 0.8
cell_line_val_ratio   = 0.1
cell_line_test_ratio  = 0.1

drug_train_ratio = 0.8
drug_val_ratio   = 0.1
drug_test_ratio  = 0.1

# Cell line 분할
train_cell_lines, temp_cell_lines = train_test_split(
    cell_lines,
    test_size=(1 - cell_line_train_ratio),
    random_state=42
)

val_cell_lines, test_cell_lines = train_test_split(
    temp_cell_lines,
    test_size=(cell_line_test_ratio / (cell_line_val_ratio + cell_line_test_ratio)),
    random_state=42
)

# Drug 분할
train_drugs, temp_drugs = train_test_split(
    drugs,
    test_size=(1 - drug_train_ratio),
    random_state=42
)

val_drugs, test_drugs = train_test_split(
    temp_drugs,
    test_size=(drug_test_ratio / (drug_val_ratio + drug_test_ratio)),
    random_state=42
)

print(f"Train Cell lines: {len(train_cell_lines)}, Val Cell lines: {len(val_cell_lines)}, Test Cell lines: {len(test_cell_lines)}")
print(f"Train Drugs: {len(train_drugs)}, Val Drugs: {len(val_drugs)}, Test Drugs: {len(test_drugs)}")

train_indices = list(product(train_cell_lines, train_drugs))
val_indices   = list(product(val_cell_lines,   val_drugs))
test_indices  = list(product(test_cell_lines,  test_drugs))

print(f"Train samples: {len(train_indices)}")
print(f"Val samples:   {len(val_indices)}")
print(f"Test samples:  {len(test_indices)}")

train_gene_embeddings, train_drug_graphs, train_drug_embeddings, train_labels = filter_data(
    train_indices, gene_embeddings, drug_graph_dict, drug_embeddings, labels_dict
)
val_gene_embeddings, val_drug_graphs, val_drug_embeddings, val_labels = filter_data(
    val_indices, gene_embeddings, drug_graph_dict, drug_embeddings, labels_dict
)
test_gene_embeddings, test_drug_graphs, test_drug_embeddings, test_labels = filter_data(
    test_indices, gene_embeddings, drug_graph_dict, drug_embeddings, labels_dict
)

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

print("Double-cold-split : Datasets saved to .pt files.")
