import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from scipy.stats import spearmanr, pearsonr
from sklearn.metrics import mean_squared_error
import numpy as np
from utils import plot_predictions

import logging
from tqdm import tqdm
import os
from datetime import datetime

from dataset import DrugResponseDataset, collate_fn
from model import DrugResponseModel
from utils import set_seed
MAX_GENE_SLOTS = 218  # 실제 사용하는 값으로 설정 (이전 로그 기반)
MAX_DRUG_SUBSTRUCTURES = 17 # 실제 사용하는 값으로 설정 (이전 로그 기반, collate_fn에서 패딩 후의 크기)

# Configuration
config = {
    'device': torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
    'batch_size': 32,
    'checkpoint_path': './checkpoints/20250521_14/best_model.pth',
    'test_data_path': './dataset_full_CV_6&7/test_dataset.pt',
}

# Model Parameters
GENE_LAYER_EMBEDDING_DIM = 64 # input dim
SUBSTRUCTURE_LAYER_EMBEDDING_DIM = 64 # input dim
CROSS_ATTN_EMBEDDING_DIM = 64
FINAL_EMBEDDING_DIM = 128
HIDDEN_DIM = 32
DEPTH = 2

BATCH_SIZE = config['batch_size']
FILE_NAME = f"test_{datetime.now().strftime('%Y%m%d_%H')}"
RESULT_DIR = f"results/{FILE_NAME}"
os.makedirs(RESULT_DIR, exist_ok=True)


# Logger 설정
log_filename = f"log/test/{FILE_NAME}.log"
os.makedirs(os.path.dirname(log_filename), exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_filename),
        logging.StreamHandler()
    ]
)
logging.info("Starting regression test script.")
set_seed(42)

# 1. Load Test Dataset
test_data = torch.load(config['test_data_path'], weights_only=False)
test_dataset = DrugResponseDataset(
    gene_embeddings=test_data['gene_embeddings'],
    drug_embeddings=test_data['drug_embeddings'],
    drug_masks=test_data['drug_masks'],
    labels=test_data['labels'],
    sample_indices=test_data['sample_indices']
)
test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False, collate_fn=collate_fn)

# 2. Load Pathway Info
pathway_masks = torch.load('./input/pathway_mask.pt')

# 3. Initialize Model
model = DrugResponseModel(
    pathway_masks=pathway_masks,
    gene_layer_dim=GENE_LAYER_EMBEDDING_DIM,
    substructure_layer_dim=SUBSTRUCTURE_LAYER_EMBEDDING_DIM,
    cross_attn_dim=CROSS_ATTN_EMBEDDING_DIM,
    final_dim=FINAL_EMBEDDING_DIM,
    max_gene_slots=MAX_GENE_SLOTS, # <<< 전달
    max_drug_substructures=MAX_DRUG_SUBSTRUCTURES # <<< 전달
)

# Load Checkpoint
checkpoint = torch.load(config['checkpoint_path'], map_location=config['device'])
model.load_state_dict(checkpoint['model_state_dict'])
model = model.to(config['device'])
model.eval()
logging.info("Model loaded and ready for regression evaluation.")

criterion = nn.MSELoss()

# 4. Test Loop
def test_model():
    total_loss = 0.0
    actuals, predictions, sample_indices_all = [], [], []

    # attention 저장 디렉토리
    attn_dir = os.path.join(RESULT_DIR, "attention_weights")
    os.makedirs(attn_dir, exist_ok=True)

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(test_loader, desc="Testing")):
            gene_embeddings = batch['gene_embeddings'].to(config['device'])
            drug_embeddings = batch['drug_embeddings'].to(config['device'])
            drug_masks = batch['drug_masks'].to(config['device'])
            labels = batch['labels'].to(config['device'])
            sample_indices = batch['sample_indices']

            outputs, gene2sub_weights, sub2gene_weights, final_pathway_embedding, final_drug_embedding = model(
                gene_embeddings, drug_embeddings, drug_masks)
            outputs = outputs.squeeze(dim=-1)

            loss = criterion(outputs, labels)
            total_loss += loss.item()

            # 결과 저장
            actuals.extend(labels.cpu().numpy())
            predictions.extend(outputs.cpu().numpy())
            sample_indices_all.extend(sample_indices)

            # # attention weight 저장
            # torch.save(gene2sub_weights.cpu(), os.path.join(attn_dir, f"B{batch_idx}_gene2sub.pt"))
            # torch.save(sub2gene_weights.cpu(), os.path.join(attn_dir, f"B{batch_idx}_sub2gene.pt"))
            # torch.save(final_pathway_embedding.cpu(), os.path.join(attn_dir, f"B{batch_idx}_pathway.pt"))
            # torch.save(final_drug_embedding.cpu(), os.path.join(attn_dir, f"B{batch_idx}_drug.pt"))
            # torch.save(sample_indices, os.path.join(attn_dir, f"B{batch_idx}_samples.pt"))


    actuals = np.array(actuals)
    predictions = np.array(predictions)

    rmse = np.sqrt(mean_squared_error(actuals, predictions))
    pcc, _ = pearsonr(actuals, predictions)
    scc, _ = spearmanr(actuals, predictions)
    test_loss = total_loss / len(test_loader)

    logging.info(f"Test Loss (MSE): {test_loss:.4f}")
    logging.info(f"Test RMSE: {rmse:.4f}")
    logging.info(f"Test Pearson Correlation (PCC): {pcc:.4f}")
    logging.info(f"Test Spearman Correlation (SCC): {scc:.4f}")

    # 결과 저장
    torch.save({
        "actuals": actuals,
        "predictions": predictions,
        "drug_labels": sample_indices_all
    }, os.path.join(RESULT_DIR, "test_results.pt"))
    logging.info(f"Test results saved to {os.path.join(RESULT_DIR, 'test_results.pt')}")

    # RMSE 플랏 저장
    plot_path = os.path.join(RESULT_DIR, "rmse_scatter_plot.png")
    plot_predictions(actuals, predictions, plot_path)
    logging.info(f"RMSE scatter plot saved to {plot_path}")


test_model()