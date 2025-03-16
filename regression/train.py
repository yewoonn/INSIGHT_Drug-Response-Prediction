import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np

import logging
import time
from tqdm import tqdm
from datetime import datetime
import os

from dataset import DrugResponseDataset, collate_fn
from model import DrugResponseModel
from utils import plot_statics, set_seed

os.environ['TZ'] = 'Asia/Seoul'
time.tzset()  # Unix 환경에서 적용
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

# Configuration
config = {
    'device': torch.device("cuda:1" if torch.cuda.is_available() else "cpu"),
    'batch_size': 16,
    'is_differ' : True,
    'depth' : 2,
    'learning_rate': 0.001,
    'weight_decay': 0.1,
    'num_epochs': 1,
    'checkpoint_dir': './checkpoints', # ckpt 디렉토리
    'plot_dir': './plots',
    'log_interval': 1, # batch 별 log 출력 간격
    'save_interval': 10, # ckpt 저장할 epoch 간격
}

NUM_CELL_LINES = 965 
NUM_PATHWAYS = 314
NUM_GENES = 264 # Max
NUM_DRUGS = 270
NUM_SUBSTRUCTURES = 17 # Max

GENE_EMBEDDING_DIM = 8
SUBSTRUCTURE_EMBEDDING_DIM = 768
EMBEDDING_DIM = 8
HIDDEN_DIM = 8
FINAL_DIM = 8
OUTPUT_DIM = 1

BATCH_SIZE = config['batch_size']
IS_DIFFER = config['is_differ']
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
    f"  NUM_GENES: {NUM_GENES} "
    f"  NUM_DRUGS: {NUM_DRUGS} "
    f"  NUM_SUBSTRUCTURES: {NUM_SUBSTRUCTURES}")
logging.info(
    "Embedding Dimensions and Hidden Layers: "
    f"  GENE_EMBEDDING_DIM: {GENE_EMBEDDING_DIM} "
    f"  SUBSTRUCTURE_EMBEDDING_DIM: {SUBSTRUCTURE_EMBEDDING_DIM} "
    f"  HIDDEN_DIM: {HIDDEN_DIM} "
    f"  FINAL_DIM: {FINAL_DIM} "
    f"  OUTPUT_DIM: {OUTPUT_DIM}")
logging.info(
    "Training Configuration: "
    f"  BATCH_SIZE: {BATCH_SIZE} "
    f"  IS_DIFFER: {IS_DIFFER} "
    f"  DEPTH: {DEPTH}")
logging.info(
    "Other Configuration: "
    f"  Learning Rate: {config['learning_rate']} "
    f"  Weight Decay: {config['weight_decay']} "
    f"  Number of Epochs: {config['num_epochs']} "
    f"  Device: {config['device']} "
    f"  Checkpoint Directory: {config['checkpoint_dir']} "
    f"  Plot Directory: {config['plot_dir']} "
    f"  Log Interval: {config['log_interval']} "
    f"  Save Interval: {config['save_interval']}"
)

logging.info(f"CUDA is available: {torch.cuda.is_available()}")

# 1. Data Loader
train_data = torch.load('./dataset/train_dataset.pt')
train_dataset = DrugResponseDataset(
    gene_embeddings=train_data['gene_embeddings'],
    drug_embeddings=train_data['drug_embeddings'],
    drug_graphs=train_data['drug_graphs'],
    drug_masks = train_data['drug_masks'],
    labels=train_data['labels'],
    sample_indices=train_data['sample_indices'],
)

val_data = torch.load('./dataset/val_dataset.pt')
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

pathway_graphs = torch.load('./input/pathway_graphs.pt')
pathway_masks = torch.load('./input/pathway_mask.pt')

# 2. Model Initialization
model = DrugResponseModel(
    pathway_graphs = pathway_graphs, 
    pathway_masks = pathway_masks,
    num_pathways = NUM_PATHWAYS, 
    num_genes = NUM_GENES, 
    num_substructures = NUM_SUBSTRUCTURES, 
    gene_dim = GENE_EMBEDDING_DIM, 
    substructure_dim = SUBSTRUCTURE_EMBEDDING_DIM, 
    embedding_dim = EMBEDDING_DIM,
    hidden_dim = HIDDEN_DIM, 
    final_dim = FINAL_DIM, 
    output_dim = OUTPUT_DIM,
    batch_size = BATCH_SIZE, 
    is_differ = IS_DIFFER, 
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
    weight_decay=config['weight_decay']
)
logging.info(f"Model initialized on device: {config['device']}")

scheduler = ReduceLROnPlateau(
    optimizer, 
    mode='min',  # Loss가 최소화될 때 반응
    factor=0.2,  
    patience=5,  # 3 에폭 동안 Loss 개선 없을 경우
    verbose=True
)

# # 저장할 샘플
# target_samples = {
#     ('DATA.684052', 'BORTEZOMIB'),
#     ('DATA.688001', 'BORTEZOMIB'),
#     ('DATA.684062', 'BORTEZOMIB'),
#     ('DATA.688023', 'BORTEZOMIB'),
#     ('DATA.688001', 'PACLITAXEL'),
#     ('DATA.684052', 'PACLITAXEL'),
#     ('DATA.684062', 'PACLITAXEL'),
#     ('DATA.688023', 'PACLITAXEL'),
# }

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

    # Epoch
    if epoch % SAVE_INTERVALS == 0:
        save_dir = f"{attn_dir}/epoch_{epoch}"
        os.makedirs(save_dir, exist_ok=True)
        torch.save(final_pathway_embedding.detach().cpu(), f"{save_dir}/B{batch_idx}_pathway_embedding.pt")
        torch.save(final_drug_embedding.detach().cpu(), f"{save_dir}/B{batch_idx}_drug_embedding.pt")
        torch.save(sample_indices, f"{save_dir}/B{batch_idx}_samples.pt")

        # for i, sample in enumerate(sample_indices):
        #     if sample in target_samples:  # Check if sample is in the target list
        #         sample_save_dir = f"{save_dir}/sample_{sample[0]}_{sample[1]}"
        #         os.makedirs(sample_save_dir, exist_ok=True)

        #         torch.save(gene2sub_weights[i].detach().cpu(), f"{sample_save_dir}/gene2sub.pt")
        #         torch.save(sub2gene_weights[i].detach().cpu(), f"{sample_save_dir}/sub2gene.pt")
        #         torch.save(labels[i].detach().cpu(), f"{sample_save_dir}/labels.pt")

        #         logging.info(
        #             f"Epoch {epoch}, Batch {batch_idx}: Saved Gene2Sub, Sub2Gene Attention Weights, "
        #             f"Pathway & Drug Embeddings, Probs, and Labels for sample {sample} to {sample_save_dir}."
        #         )

    return loss, rmse, sample_indices

# 4. Training Loop
train_losses, val_losses = [], []
train_rmses, val_rmses = [], []

# 결과 저장 폴더 생성
result_dir = os.path.join("results", FILE_NAME)
os.makedirs(result_dir, exist_ok=True)
os.makedirs(chpt_dir, exist_ok=True)  # Checkpoints 저장할 디렉토리 생성

# with prof:
for epoch in range(config['num_epochs']):
    epoch_start = time.time()
    logging.info(f"Epoch [{epoch+1}/{config['num_epochs']}] started.")

    train_actuals, train_predictions, train_drug_labels = [], [], []
    val_actuals, val_predictions, val_drug_labels = [], [], []

    # Training Phase
    model.train()
    total_train_loss, total_train_rmse = 0, 0

    for batch_idx, batch in enumerate(tqdm(train_loader, desc=f"Train Epoch {epoch+1}")):
        optimizer.zero_grad()
        loss, rmse, outputs  = process_batch(batch_idx, batch, epoch+1, model, criterion, config['device'])
        loss.backward()
        optimizer.step()

        total_train_loss += loss.item()
        total_train_rmse += rmse

        if batch_idx % config['log_interval'] == 0:
            logging.info(f"Batch {batch_idx+1}: Loss: {loss.item():.4f}, ")   

     # 예측값 저장
        labels = batch['labels'].cpu().numpy()
        if isinstance(outputs, torch.Tensor):
            outputs = outputs.detach().cpu().numpy()
        else:
            outputs = np.array(outputs)
        drug_labels = [sample[1] for sample in batch['sample_indices']]

        train_actuals.extend(labels)
        train_predictions.extend(outputs)
        train_drug_labels.extend(drug_labels)

    train_loss = total_train_loss / len(train_loader)
    train_rmse = total_train_rmse / len(train_loader)  # RMSE 평균
    train_losses.append(train_loss)
    train_rmses.append(train_rmse)

     # Validation Phase
    model.eval()
    total_val_loss, total_val_rmse = 0, 0

    with torch.no_grad():
        for batch in tqdm(val_loader, desc=f"Validation Epoch {epoch+1}"):
            loss, rmse, outputs = process_batch(batch_idx, batch, epoch+1, model, criterion, config['device'])

            total_val_loss += loss.item()
            total_val_rmse += rmse

            labels = batch['labels'].cpu().numpy()
            if isinstance(outputs, torch.Tensor):
                outputs = outputs.detach().cpu().numpy()
            else:
                outputs = np.array(outputs)
            drug_labels = [sample[1] for sample in batch['sample_indices']]

            val_actuals.extend(labels)
            val_predictions.extend(outputs)
            val_drug_labels.extend(drug_labels)

    val_loss = total_val_loss / len(val_loader)
    val_rmse = total_val_rmse / len(val_loader)
    val_losses.append(val_loss)
    val_rmses.append(val_rmse)

    scheduler.step(val_loss)

    logging.info(f"Epoch [{epoch+1}/{config['num_epochs']}] completed. "
                 f"Train Loss: {train_loss:.4f}, Train RMSE: {train_rmse:.4f}, "
                 f"Val Loss: {val_loss:.4f}, Val RMSE: {val_rmse:.4f}\n"
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
        "drug_labels": train_drug_labels
    }, os.path.join(result_dir, f"train_results_epoch_{epoch+1}.pt"))

    torch.save({
        "actuals": val_actuals,
        "predictions": val_predictions,
        "drug_labels": val_drug_labels
    }, os.path.join(result_dir, f"val_results_epoch_{epoch+1}.pt"))

    logging.info(f"Results saved for epoch {epoch+1} in directory {result_dir}")

    plot_statics(os.path.join(result_dir, FILE_NAME), train_losses, val_losses, train_rmses, val_rmses)
    
# Loss 및 RMSE 그래프 저장
plot_statics(os.path.join(result_dir, FILE_NAME), train_losses, val_losses, train_rmses, val_rmses)
logging.info(f"Plots saved in directory {result_dir}")