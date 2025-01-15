import torch
import numpy as np
from itertools import product
import numpy as np
from sklearn.model_selection import train_test_split


# 필터링
def filter_data(sample_indices, gene_embeddings, drug_graphs, substructure_embeddings, labels_dict):
    filtered_gene_embeddings = {cell_line: gene_embeddings[cell_line] for cell_line, _ in sample_indices}
    filtered_drug_graphs = {drug: drug_graphs[drug] for _, drug in sample_indices}
    filtered_substructure_embeddings = {
        drug: substructure_embeddings[drug] for _, drug in sample_indices
    }
    filtered_labels = {(cell_line, drug): labels_dict[(cell_line, drug)] for cell_line, drug in sample_indices}
    return filtered_gene_embeddings, filtered_drug_graphs, filtered_substructure_embeddings, filtered_labels

# =================================================================================================

num_cell_lines = 1280
num_pathways = 312
num_genes = 3848
num_drugs = 78
num_substructures = 194

# Gene Embedding
gene_embeddings = torch.load('../input/0_gene_embeddings.pt')

# Pathway
pathways = torch.load("../input/0_pathway_graph.pt")
global_ids_list = [data["global_ids"] for data in pathways]
global_ids_dict = {index: global_ids for index, global_ids in enumerate(global_ids_list)}
torch.save(global_ids_dict, "./pathway_genes_dict.pt")

# Drug Embedding
saved_embeddings = np.load('../input/0_drug_embeddings.npz')
substructure_embeddings = {
    key: torch.tensor(saved_embeddings[key], dtype=torch.Int)
    for key in saved_embeddings.keys()
}
saved_embeddings.close()
drug_graph_dict = torch.load('../input/0_drug_graph_dict.pt')

# Drug Label
labels_dict = torch.load('../input/0_drug_label_dict.pt')

cell_lines = list(gene_embeddings.keys())
drugs = list(substructure_embeddings.keys())
sample_indices = list(product(cell_lines, drugs)) 

print("cell_lines : ", len(cell_lines), "drugs : ", len(drugs))

# 데이터셋 분할 비율 (0.7, 0.15, 0.15)
train_ratio = 0.8
val_ratio = 0.1

# Sample indices 분할
train_indices, temp_indices = train_test_split(sample_indices, test_size=(1 - train_ratio), random_state=42)
val_indices, test_indices = train_test_split(temp_indices, test_size=(1 - val_ratio / (1 - train_ratio)), random_state=42)

# 필터링
train_gene_embeddings, train_drug_graphs, train_substructure_embeddings, train_labels = filter_data(
    train_indices, gene_embeddings, drug_graph_dict, substructure_embeddings, labels_dict
)

val_gene_embeddings, val_drug_graphs, val_substructure_embeddings, val_labels = filter_data(
    val_indices, gene_embeddings, drug_graph_dict, substructure_embeddings, labels_dict
)

test_gene_embeddings, test_drug_graphs, test_substructure_embeddings, test_labels = filter_data(
    test_indices, gene_embeddings, drug_graph_dict, substructure_embeddings, labels_dict
)

# Save train data
torch.save({
    'gene_embeddings': train_gene_embeddings,
    'substructure_embeddings': train_substructure_embeddings,
    'drug_graphs': train_drug_graphs,
    'labels': train_labels,
    'sample_indices': train_indices,
}, './dataset/train_dataset.pt')

# Save validation data
torch.save({
    'gene_embeddings': val_gene_embeddings,
    'substructure_embeddings': val_substructure_embeddings,
    'drug_graphs': val_drug_graphs,
    'labels': val_labels,
    'sample_indices': val_indices,
}, './dataset/val_dataset.pt')

# Save test data
torch.save({
    'gene_embeddings': test_gene_embeddings,
    'substructure_embeddings': test_substructure_embeddings,
    'drug_graphs': test_drug_graphs,
    'labels': test_labels,
    'sample_indices': test_indices,
}, './dataset/test_dataset.pt')

print("Datasets saved to .pt files.")
