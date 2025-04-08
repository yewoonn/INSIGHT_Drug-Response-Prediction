import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import random
from scipy.stats import spearmanr, pearsonr

import logging
import time
from tqdm import tqdm
from datetime import datetime
import os

from dataset import DrugResponseDataset, collate_fn
from model import DrugResponseModel
from utils import plot_statics, set_seed

os.environ['TZ'] = 'Asia/Seoul'
time.tzset()
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

# Configuration
config = {
    'device': torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
    'batch_size': 16,
    'depth' : 2,
    'learning_rate': 0.005,
    'num_epochs': 10,
    'checkpoint_dir': './checkpoints', # ckpt 디렉토리
    'plot_dir': './plots',
    'log_interval': 10, # batch 별 log 출력 간격
    'save_interval': 1, # ckpt 저장할 epoch 간격
}

NUM_CELL_LINES = 964
NUM_PATHWAYS = 314
NUM_MAX_GENES = 264 # Max
NUM_DRUGS = 270
NUM_MAX_SUBSTRUCTURES = 17 # Max

GENE_LAYER_EMBEDDING_DIM = 8 # input dim
SUBSTRUCTURE_LAYER_EMBEDDING_DIM = 768 # input dim
CROSS_ATTN_EMBEDDING_DIM = 8 
GRAPH_EMBEDDING_DIM = 64
OUTPUT_DIM = 1
FINAL_EMBEDDING_DIM = 128
HIDDEN_DIM = 64

BATCH_SIZE = config['batch_size']
DEPTH = config['depth']

FILE_NAME = datetime.now().strftime('%Y%m%d_%H')
DEVICE = config['device']
SAVE_INTERVALS = config['save_interval']

log_filename = f"log/train/{FILE_NAME}.log"

chpt_dir = f"{config['checkpoint_dir']}/{FILE_NAME}"
os.makedirs(chpt_dir, exist_ok=True)

attn_dir = f"weights/{FILE_NAME}"
os.makedirs(attn_dir, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_filename),
        logging.StreamHandler()
    ]
)

logging.info(
    "Model Configuration and Parameters: "
    f"  NUM_CELL_LINES: {NUM_CELL_LINES} "
    f"  NUM_PATHWAYS: {NUM_PATHWAYS} "
    f"  NUM_GENES: {NUM_MAX_GENES} "
    f"  NUM_DRUGS: {NUM_DRUGS} "
    f"  NUM_SUBSTRUCTURES: {NUM_MAX_SUBSTRUCTURES}")
logging.info(
    "Embedding Dimensions and Hidden Layers: "
    f"  GENE_EMBEDDING_DIM: {GENE_LAYER_EMBEDDING_DIM} "
    f"  SUBSTRUCTURE_EMBEDDING_DIM: {SUBSTRUCTURE_LAYER_EMBEDDING_DIM} "
    f"  HIDDEN_DIM: {HIDDEN_DIM} "
    f"  FINAL_DIM: {FINAL_EMBEDDING_DIM} "
    f"  OUTPUT_DIM: {OUTPUT_DIM}")
logging.info(
    "Training Configuration: "
    f"  BATCH_SIZE: {BATCH_SIZE} "
    f"  DEPTH: {DEPTH}")
logging.info(
    "Other Configuration: "
    f"  Learning Rate: {config['learning_rate']} "
    f"  Number of Epochs: {config['num_epochs']} "
    f"  Device: {config['device']} "
    f"  Checkpoint Directory: {config['checkpoint_dir']} "
    f"  Plot Directory: {config['plot_dir']} "
    f"  Log Interval: {config['log_interval']} "
    f"  Save Interval: {config['save_interval']}"
)

logging.info(f"CUDA is available: {torch.cuda.is_available()}")

# 1. Data Loader
train_data = torch.load('./dataset_10_0307_LN/train_dataset.pt')
train_dataset = DrugResponseDataset(
    gene_embeddings=train_data['gene_embeddings'],
    drug_embeddings=train_data['drug_embeddings'],
    drug_graphs=train_data['drug_graphs'],
    drug_masks = train_data['drug_masks'],
    labels=train_data['labels'],
    sample_indices=train_data['sample_indices'],
)

val_data = torch.load('./dataset_10_0307_LN/val_dataset.pt')
val_dataset = DrugResponseDataset(
    gene_embeddings=val_data['gene_embeddings'],
    drug_embeddings=val_data['drug_embeddings'],
    drug_graphs=val_data['drug_graphs'],
    drug_masks = val_data['drug_masks'],
    labels=val_data['labels'],
    sample_indices=val_data['sample_indices'],
)

# Seed 설정
seed = 42
set_seed(42)

train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False, collate_fn=collate_fn)

pathway_graphs = torch.load('./input_0307/pathway_graphs.pt')
pathway_masks = torch.load('./input_0307/pathway_mask.pt')

# 2. Model Initialization
model = DrugResponseModel(
    pathway_graphs = pathway_graphs, 
    pathway_masks = pathway_masks,
    num_pathways = NUM_PATHWAYS,
    gene_layer_dim = GENE_LAYER_EMBEDDING_DIM, 
    substructure_layer_dim = SUBSTRUCTURE_LAYER_EMBEDDING_DIM, 
    cross_attn_dim = CROSS_ATTN_EMBEDDING_DIM, 
    graph_dim = GRAPH_EMBEDDING_DIM, 
    final_dim = FINAL_EMBEDDING_DIM, 
    output_dim = OUTPUT_DIM,
    batch_size = BATCH_SIZE, 
    depth = DEPTH, 
    save_intervals = SAVE_INTERVALS, 
    file_name = FILE_NAME,
    device = DEVICE
)

logging.info(f"Initial GPU memory usage (before moving model to device): "
             f"{torch.cuda.memory_allocated(config['device']) / 1e6:.2f} MB allocated, "
             f"{torch.cuda.memory_reserved(config['device']) / 1e6:.2f} MB reserved.")

model = model.to(config['device'])
criterion = nn.MSELoss()
optimizer = optim.Adam(
    model.parameters(),
    lr=config['learning_rate'],
)
logging.info(f"Model initialized on device: {config['device']}")

# 3. Helper Function
def process_batch(batch_idx, batch, epoch, model, criterion, device):
    gene_embeddings = batch['gene_embeddings'].to(device) # [Batch, Pathway_num, Max_Gene]
    drug_embeddings = batch['drug_embeddings'].to(device) # [Batch, Max_Sub]
    drug_graphs = batch['drug_graphs'].to(device) # DataBatch
    drug_masks = batch['drug_masks'].to(device) # [Batch, Max_Sub]
    labels = batch['labels'].to(device) # [Batch]
    sample_indices = batch['sample_indices']

    outputs, gene2sub_weights, sub2gene_weights, final_pathway_embedding, final_drug_embedding = model(gene_embeddings, drug_embeddings, drug_graphs, drug_masks)    
    outputs = outputs.squeeze(dim=-1) 
    loss = criterion(outputs, labels) 

    rmse = torch.sqrt(loss).item()  # RMSE 계산

    if epoch % SAVE_INTERVALS == 0:
        save_dir = f"{attn_dir}/epoch_{epoch}"
        os.makedirs(save_dir, exist_ok=True)
        torch.save(final_pathway_embedding.detach().cpu(), f"{save_dir}/B{batch_idx}_pathway_embedding.pt")
        torch.save(final_drug_embedding.detach().cpu(), f"{save_dir}/B{batch_idx}_drug_embedding.pt")
        torch.save(sample_indices, f"{save_dir}/B{batch_idx}_samples.pt") # (세포주, 약물)

    return outputs, loss, rmse, sample_indices, labels

# 4. Training Loop
train_losses, val_losses = [], []
train_rmses, val_rmses = [], []

# 결과 저장 폴더 생성
result_dir = os.path.join("results", FILE_NAME)
os.makedirs(result_dir, exist_ok=True)
os.makedirs(chpt_dir, exist_ok=True)

start_time = time.time()

for epoch in range(config['num_epochs']):
    epoch_start = time.time()
    logging.info(f"Epoch [{epoch+1}/{config['num_epochs']}] started.")

    train_actuals, train_predictions, train_samples = [], [], []
    val_actuals, val_predictions, val_samples = [], [], []

    # Training Phase
    model.train()
    total_train_loss, total_train_rmse = 0, 0

    for batch_idx, batch in enumerate(tqdm(train_loader, desc=f"Train Epoch {epoch+1}")):
        optimizer.zero_grad()
        outputs, loss, rmse, sample_indices, labels  = process_batch(batch_idx, batch, epoch+1, model, criterion, config['device'])
        loss.backward()
        optimizer.step()

        total_train_loss += loss.item()
        total_train_rmse += rmse

        if batch_idx % config['log_interval'] == 0:
                logging.info(f"Batch {batch_idx+1}: Loss: {loss.item():.4f}, ")   

         # 예측값 저장
        train_actuals.extend(labels) # 실제값
        train_predictions.extend(outputs) # 예측값
        train_samples.extend(sample_indices) # 라벨

    train_loss = total_train_loss / len(train_loader)
    train_rmse = total_train_rmse / len(train_loader)  # RMSE 평균
    train_losses.append(train_loss)
    train_rmses.append(train_rmse)

    # Validation Phase
    model.eval()
    total_val_loss, total_val_rmse = 0, 0

    with torch.no_grad():
        for batch in tqdm(val_loader, desc=f"Validation Epoch {epoch+1}"):
            outputs, loss, rmse, sample_indices, labels = process_batch(batch_idx, batch, epoch+1, model, criterion, config['device'])
            total_val_loss += loss.item()
            total_val_rmse += rmse
            val_actuals.extend(labels)
            val_predictions.extend(outputs)
            val_samples.extend(sample_indices)

    val_loss = total_val_loss / len(val_loader)
    val_rmse = total_val_rmse / len(val_loader)
    val_losses.append(val_loss)
    val_rmses.append(val_rmse)

        # SCC/PCC 계산
    train_actuals_np = torch.stack(train_actuals).cpu().numpy()
    train_predictions_np = torch.stack(train_predictions).detach().cpu().numpy()
    val_actuals_np = torch.stack(val_actuals).cpu().numpy()
    val_predictions_np = torch.stack(val_predictions).detach().cpu().numpy()

    train_pcc = pearsonr(train_actuals_np, train_predictions_np)[0]
    train_scc = spearmanr(train_actuals_np, train_predictions_np)[0]
    val_pcc = pearsonr(val_actuals_np, val_predictions_np)[0]
    val_scc = spearmanr(val_actuals_np, val_predictions_np)[0]

    # scheduler.step(val_loss)
    logging.info(f"Epoch [{epoch+1}/{config['num_epochs']}] completed. \n"
                 f"Train Loss: {train_loss:.4f}, Train RMSE: {train_rmse:.4f}, Train PCC: {train_pcc:.4f}, Train SCC: {train_scc:.4f},\n"
                 f"Val Loss: {val_loss:.4f}, Val RMSE: {val_rmse:.4f} Val PCC: {val_pcc:.4f}, Val SCC: {val_scc:.4f} \n"
    )

    # Checkpoints 저장 (매 Epoch 저장)
    checkpoint_path = f"{chpt_dir}/epoch_{epoch+1}.pth"
    torch.save({
        'epoch': epoch + 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': train_loss,
        'val_loss': val_loss,
    }, checkpoint_path)
    logging.info(f"Checkpoint saved at {checkpoint_path}")

        # 결과 저장 (에폭마다 다른 파일명 사용)
    torch.save({
        "actuals": train_actuals,
        "predictions": train_predictions,
        "drug_labels": train_samples
    }, os.path.join(result_dir, f"train_results_epoch_{epoch+1}.pt"))

    torch.save({
        "actuals": val_actuals,
        "predictions": val_predictions,
        "drug_labels": val_samples
    }, os.path.join(result_dir, f"val_results_epoch_{epoch+1}.pt"))

    logging.info(f"Results saved for epoch {epoch+1} in directory {result_dir}")

    plot_statics(os.path.join(result_dir, FILE_NAME), train_losses, val_losses, train_rmses, val_rmses)
    
# Loss 및 RMSE 그래프 저장
plot_statics(os.path.join(result_dir, FILE_NAME), train_losses, val_losses, train_rmses, val_rmses)
logging.info(f"Plots saved in directory {result_dir}")