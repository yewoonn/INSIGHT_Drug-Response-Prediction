import torch
from itertools import product
from sklearn.model_selection import train_test_split


# 필터링
def filter_data(sample_indices, gene_embeddings, drug_embeddings, drug_graphs, drug_masks, labels_dict):
    filtered_gene_embeddings = {cell_line: gene_embeddings[cell_line] for cell_line, _ in sample_indices}
    filtered_drug_embeddings = {
        drug: drug_embeddings[drug] for _, drug in sample_indices
    }
    filtered_drug_graphs = {drug: drug_graphs[drug] for _, drug in sample_indices}
    filtered_drug_masks = {
        drug: drug_masks[drug] for _, drug in sample_indices
    }
    filtered_labels = {(cell_line, drug): labels_dict[(cell_line, drug)] for cell_line, drug in sample_indices}
    return filtered_gene_embeddings, filtered_drug_embeddings, filtered_drug_graphs, filtered_drug_masks, filtered_labels

# =================================================================================================

gene_embeddings = torch.load('./input_10/gene_embeddings.pt')

drug_embeddings = torch.load('./input_10/drug_embeddings.pt')
drug_graph_dict = torch.load('./input_10/drug_graph_dict.pt')
drug_masks = torch.load('./input_10/drug_mask_dict.pt')

labels_dict = torch.load('./input_10/drug_label_dict.pt')

cell_lines = list(gene_embeddings.keys())
drugs = list(drug_embeddings.keys())
print(drugs)
sample_indices = list(product(cell_lines, drugs)) 

print("cell_lines : ", len(cell_lines), "drugs : ", len(drugs))

# 데이터셋 분할 비율 (0.8, 0.1, 0.1)
train_ratio = 0.8
val_ratio = 0.1

# Sample indices 분할
train_indices, temp_indices = train_test_split(sample_indices, test_size=(1 - train_ratio), random_state=42)
val_indices, test_indices = train_test_split(temp_indices, test_size=(1 - val_ratio / (1 - train_ratio)), random_state=42)

# 필터링
train_gene_embeddings, train_drug_embeddings, train_drug_graphs, train_masks, train_labels = filter_data(
    train_indices, gene_embeddings, drug_embeddings, drug_graph_dict, drug_masks, labels_dict
)

val_gene_embeddings, val_drug_embeddings, val_drug_graphs, val_masks, val_labels = filter_data(
    val_indices, gene_embeddings, drug_embeddings, drug_graph_dict, drug_masks, labels_dict
)

test_gene_embeddings, test_drug_embeddings, test_drug_graphs, test_masks, test_labels = filter_data(
    test_indices, gene_embeddings, drug_embeddings, drug_graph_dict, drug_masks, labels_dict
)

# Save train data
torch.save({
    'gene_embeddings': train_gene_embeddings,
    'drug_embeddings': train_drug_embeddings,
    'drug_graphs': train_drug_graphs,
    'drug_masks' : train_masks,
    'labels': train_labels,
    'sample_indices': train_indices,
}, './dataset/train_dataset.pt')


# Save validation data
torch.save({
    'gene_embeddings': val_gene_embeddings,
    'drug_embeddings': val_drug_embeddings,
    'drug_graphs': val_drug_graphs,
    'drug_masks' : val_masks,
    'labels': val_labels,
    'sample_indices': val_indices,
}, './dataset/val_dataset.pt')

# Save test data
torch.save({
    'gene_embeddings': test_gene_embeddings,
    'drug_embeddings': test_drug_embeddings,
    'drug_graphs': test_drug_graphs,
    'drug_masks' : test_masks,
    'labels': test_labels,
    'sample_indices': test_indices,
}, './dataset/test_dataset.pt')

print("Datasets saved to .pt files.")
