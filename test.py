import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from scipy.stats import spearmanr, pearsonr
from sklearn.metrics import mean_squared_error
import numpy as np

import logging
from tqdm import tqdm
import os
from datetime import datetime
import yaml
import argparse

from dataset import DrugResponseDataset, collate_fn
from model import DrugResponseModel
from utils import set_seed, plot_predictions

# ========== Configuration ==========
# Parse command line
def parse_args():
    parser = argparse.ArgumentParser(description='Drug Response Prediction Testing')
    parser.add_argument('--config', type=str, default='config.yml', 
                       help='Path to configuration file (default: config.yml)')
    parser.add_argument('--checkpoint_date', type=str, required=True,
                       help='Checkpoint date folder (e.g., 20250712_21)')
    return parser.parse_args()

# Load configuration
def load_config(config_path='config.yml'):
    with open(config_path, 'r', encoding='utf-8') as file:
        config = yaml.safe_load(file)
    return config

args = parse_args()
config = load_config(args.config)

CHECKPOINT_DATE = args.checkpoint_date  # 커맨드 라인에서 입력받음

# Extract config parameters
model_config = config['model']
data_config = config['data']
training_config = config['training']

# Data paths
TEST_DATA_PATH = data_config['test_data_path']
PATHWAY_GENE_INDICES_PATH = data_config['pathway_gene_indices_path']

# Model parameters
GENE_FFN_OUTPUT_DIM = model_config['gene_ffn_output_dim']
DRUG_FFN_OUTPUT_DIM = model_config['drug_ffn_output_dim']
GENE_INPUT_DIM = model_config.get('gene_input_dim', 154)
DRUG_INPUT_DIM = model_config.get('drug_input_dim', 768)
IS_DIFFER = model_config.get('isDiffer', True)
CROSS_ATTN_EMBEDDING_DIM = model_config['cross_attn_embedding_dim']
FINAL_EMBEDDING_DIM = model_config['final_embedding_dim']
OUTPUT_DIM = model_config['output_dim']
MAX_GENE_SLOTS = model_config['max_gene_slots']

# Training parameters
BATCH_SIZE = training_config['batch_size']

DEVICE = training_config['device']
FILE_NAME = f"{CHECKPOINT_DATE}"
RESULT_DIR = f"results/{FILE_NAME}"
LOG_FILE = f"log/{FILE_NAME}/test.log"
os.makedirs(RESULT_DIR, exist_ok=True)
os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)

# ========== Logging ==========
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler()
    ]
)
logging.info("🧪 Starting 5-Fold Regression Test")
set_seed(42)

# ========== Load Test Set ==========
test_data = torch.load(TEST_DATA_PATH)

test_dataset = DrugResponseDataset(
    gene_embeddings=test_data['gene_embeddings'],
    drug_embeddings=test_data['drug_embeddings'],
    drug_substructure_embeddings=test_data['drug_substructure_embeddings'],
    drug_substructure_masks=test_data['drug_substructure_masks'],
    labels=test_data['labels'],
    sample_indices=test_data['sample_indices']
)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

# ========== Load Pathway Info ==========
pathway_gene_indices = torch.load(PATHWAY_GENE_INDICES_PATH)

# ========== Evaluation ==========
summary_stats = []
all_fold_results = {}

for fold in range(1, 6):
    logging.info(f"\n📂 Fold {fold} Evaluation Start")
    checkpoint_path = f'./checkpoints/{CHECKPOINT_DATE}/fold_{fold}/best_model.pth'
    # No need for fold-specific directories anymore

    # --- 모델 초기화 수정 ---
    model = DrugResponseModel(
        pathway_gene_indices=pathway_gene_indices,
        gene_ffn_output_dim=GENE_FFN_OUTPUT_DIM,
        drug_ffn_output_dim=DRUG_FFN_OUTPUT_DIM,
        cross_attn_dim=CROSS_ATTN_EMBEDDING_DIM,
        final_dim=FINAL_EMBEDDING_DIM,
        max_gene_slots=MAX_GENE_SLOTS,
        gene_input_dim=GENE_INPUT_DIM,
        drug_input_dim=DRUG_INPUT_DIM,
        isDiffer=IS_DIFFER,
        gene_ffn_hidden_dim=model_config['gene_ffn_hidden_dim'],
        drug_ffn_hidden_dim=model_config['drug_ffn_hidden_dim'],
        gene_ffn_dropout=model_config['gene_ffn_dropout'],
        drug_ffn_dropout=model_config['drug_ffn_dropout'],
        num_heads=model_config['num_heads'],
        depth=model_config['depth'],
        mlp_dropout=model_config['mlp_dropout'],
        final_dim_reduction_factor=model_config['final_dim_reduction_factor'],
    ).to(DEVICE)

    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    criterion = nn.MSELoss()
    actuals, predictions, sample_indices_all = [], [], []
    total_loss = 0.0

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(test_loader, desc=f"Testing Fold {fold}")):
            gene_embeddings = batch['gene_embeddings'].to(DEVICE)
            drug_embeddings = batch['drug_embeddings'].to(DEVICE)  # [B, 768]
            drug_substructure_embeddings = batch['drug_substructure_embeddings'].to(DEVICE)  # [B, L, 768]
            drug_substructure_masks = batch['drug_substructure_masks'].to(DEVICE)  # [B, L]
            labels = batch['labels'].to(DEVICE)
            sample_indices = batch['sample_indices']

            outputs, *_ = model(gene_embeddings, drug_embeddings, drug_substructure_embeddings, drug_substructure_masks)
            outputs = outputs.squeeze(dim=-1)

            loss = criterion(outputs, labels)
            total_loss += loss.item()

            actuals.extend(labels.cpu().numpy())
            predictions.extend(outputs.cpu().numpy())
            sample_indices_all.extend(sample_indices)

    # 성능 지표 계산
    actuals = np.array(actuals)
    predictions = np.array(predictions)
    rmse = np.sqrt(mean_squared_error(actuals, predictions))
    pcc, _ = pearsonr(actuals, predictions)
    scc, _ = spearmanr(actuals, predictions)
    test_loss = total_loss / len(test_loader)

    logging.info(f"✅ Fold {fold} Evaluation Complete")
    logging.info(f"Test Loss: {test_loss:.4f}, RMSE: {rmse:.4f}, PCC: {pcc:.4f}, SCC: {scc:.4f}")

    # Store fold results
    all_fold_results[f"fold_{fold}"] = {
        "actuals": actuals,
        "predictions": predictions,
        "sample_indices": sample_indices_all,
        "rmse": rmse,
        "pcc": pcc,
        "scc": scc
    }

    summary_stats.append({
        "fold": fold,
        "rmse": rmse,
        "pcc": pcc,
        "scc": scc,
        "loss": test_loss
    })

# ========== Save Consolidated Results ==========
test_results_path = os.path.join(RESULT_DIR, "test_results.pt")
torch.save(all_fold_results, test_results_path)
logging.info(f"✅ Test results saved to {test_results_path}")

# ========== Summary Save ==========
summary_log_path = os.path.join(RESULT_DIR, "summary.log")
with open(summary_log_path, "w") as f:
    for stat in summary_stats:
        f.write(f"Fold {stat['fold']} - Loss: {stat['loss']:.4f}, RMSE: {stat['rmse']:.4f}, "
                f"PCC: {stat['pcc']:.4f}, SCC: {stat['scc']:.4f}\n")

logging.info("🎯 All folds evaluated successfully. Summary saved.")

# ========== Mean ± Std Summary ==========
import numpy as np

def mean_std_str(arr):
    return f"{np.mean(arr):.4f} ± {np.std(arr):.4f}"

rmse_list = [s["rmse"] for s in summary_stats]
pcc_list = [s["pcc"] for s in summary_stats]
scc_list = [s["scc"] for s in summary_stats]
loss_list = [s["loss"] for s in summary_stats]

logging.info("📊 5-Fold Cross-Validation Summary (Mean ± Std)")
logging.info(f"RMSE: {mean_std_str(rmse_list)}")
logging.info(f"PCC:  {mean_std_str(pcc_list)}")
logging.info(f"SCC:  {mean_std_str(scc_list)}")
logging.info(f"Loss: {mean_std_str(loss_list)}")

with open(summary_log_path, "a") as f:
    f.write("\n📊 Mean ± Std:\n")
    f.write(f"RMSE: {mean_std_str(rmse_list)}\n")
    f.write(f"PCC:  {mean_std_str(pcc_list)}\n")
    f.write(f"SCC:  {mean_std_str(scc_list)}\n")
    f.write(f"Loss: {mean_std_str(loss_list)}\n")