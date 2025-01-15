import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.amp import autocast
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

import logging
from tqdm import tqdm
import os
from datetime import datetime

from dataset import DrugResponseDataset, collate_fn
from model import DrugResponseModel
from utils import AttentionLogger

# Configuration
config = {
    'device': torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
    'batch_size': 64,
    'checkpoint_path': './checkpoints/20250114_23/ckpt_epoch_10.pth', # 변경 필요
    'test_data_path': 'dataset/test_dataset.pt',
}

NUM_PATHWAYS = 312
NUM_GENES = 3848
NUM_SUBSTRUCTURES = 194
GENE_EMBEDDING_DIM = 32
SUBSTRUCTURE_EMBEDDING_DIM = 32
HIDDEN_DIM = 32
FINAL_DIM = 16
OUTPUT_DIM = 1
BATCH_SIZE = config['batch_size']

# Logger 설정
log_filename = f"log/test/{datetime.now().strftime('%Y%m%d_%H')}.log"
os.makedirs(os.path.dirname(log_filename), exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_filename),
        logging.StreamHandler()
    ]
)

logging.info("Starting test script.")

# 1. Load Test Dataset
test_data = torch.load(config['test_data_path'])
test_dataset = DrugResponseDataset(
    gene_embeddings=test_data['gene_embeddings'],
    drug_embeddings=test_data['drug_embeddings'],
    drug_graphs=test_data['drug_graphs'],
    labels=test_data['labels'],
    sample_indices=test_data['sample_indices']
)

test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False, collate_fn=collate_fn)

# 2. Load Pathway Information
pathway_genes_dict = torch.load('./input/0_pathway_genes_dict.pt')
pathway_graphs = torch.load('./input/0_pathway_graph.pt')

# 3. Model Initialization
dummy_logger = AttentionLogger()  # 또는 필요 없을 경우 None 사용

model = DrugResponseModel(
    num_pathways=NUM_PATHWAYS,
    pathway_graphs=pathway_graphs,
    pathway_genes_dict=pathway_genes_dict,
    num_genes=NUM_GENES,
    num_substructures=NUM_SUBSTRUCTURES,
    gene_dim=GENE_EMBEDDING_DIM,
    substructure_dim=SUBSTRUCTURE_EMBEDDING_DIM,
    hidden_dim=HIDDEN_DIM,
    final_dim=FINAL_DIM,
    output_dim=OUTPUT_DIM,
    batch_size=BATCH_SIZE,
    is_differ=True,  # 또는 False, 테스트에 적합한 값으로 설정
    depth=1,         # 필요한 값으로 설정
    save_intervals=10,  # 로깅 간격
    save_pathways=[0],  # 테스트용 경로 ID
    file_name="test",   # 테스트 파일 이름
    attn_logger=dummy_logger  # 또는 None
)

checkpoint = torch.load(config['checkpoint_path'], map_location=config['device'])
model.load_state_dict(checkpoint['model_state_dict'])
model = model.to(config['device'])

logging.info("Model loaded successfully.")

criterion = nn.BCEWithLogitsLoss()

# 4. Test Loop
def test_model():
    model.eval()
    total_test_loss, correct_preds, total_samples = 0, 0, 0
    y_true, y_pred = [], []

    with torch.no_grad():
        with autocast(device_type="cuda" if config['device'].type == "cuda" else "cpu"):
            for batch in tqdm(test_loader, desc="Testing"):
                gene_embeddings = batch['gene_embeddings'].to(config['device'])
                substructure_embeddings = batch['substructure_embeddings'].to(config['device'])
                drug_graphs = batch['drug_graphs'].to(config['device'])
                labels = batch['labels'].to(config['device'])
                sample_indices = batch['sample_indices']

                outputs = model(gene_embeddings, substructure_embeddings, drug_graphs, -1, sample_indices)
                outputs = outputs.squeeze(dim=-1)
                loss = criterion(outputs, labels)

                total_test_loss += loss.item()

                preds = (torch.sigmoid(outputs) > 0.5).long()
                correct_preds += (preds == labels).sum().item()
                total_samples += labels.size(0)

                y_true.extend(labels.cpu().numpy())
                y_pred.extend(preds.cpu().numpy())

    test_loss = total_test_loss / len(test_loader)
    test_accuracy = accuracy_score(y_true, y_pred)
    test_precision = precision_score(y_true, y_pred)
    test_recall = recall_score(y_true, y_pred)
    test_f1 = f1_score(y_true, y_pred)

    logging.info(f"Test Loss: {test_loss:.4f}")
    logging.info(f"Test Accuracy: {test_accuracy:.4f}")
    logging.info(f"Test Precision: {test_precision:.4f}")
    logging.info(f"Test Recall: {test_recall:.4f}")
    logging.info(f"Test F1 Score: {test_f1:.4f}")

test_model()
