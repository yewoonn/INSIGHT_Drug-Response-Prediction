import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.amp import GradScaler, autocast

import logging
import time
from tqdm import tqdm
from datetime import datetime
import os

from dataset import DrugResponseDataset, collate_fn
from model import DrugResponseModel
from utils import plot_statics, clear_cache

os.environ['TZ'] = 'Asia/Seoul'
time.tzset()  # Unix 환경에서 적용
# os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:1024' # OR 512
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

# Configuration
config = {
    'device': torch.device("cuda:3" if torch.cuda.is_available() else "cpu"),
    'batch_size': 2,
    'is_differ' : True,
    'depth' : 2,
    'learning_rate': 0.001,
    'num_epochs': 10,
    'checkpoint_dir': './checkpoints', # ckpt 디렉토리
    'plot_dir': './plots',
    'log_interval': 10, # batch 별 log 출력 간격
    'save_interval': 1, # ckpt 저장할 epoch 간격
}

NUM_CELL_LINES = 1280
NUM_PATHWAYS = 312
NUM_GENES = 3848
NUM_DRUGS = 78
NUM_SUBSTRUCTURES = 194

GENE_EMBEDDING_DIM = 32
SUBSTRUCTURE_EMBEDDING_DIM = 32
HIDDEN_DIM = 32
FINAL_DIM = 16
OUTPUT_DIM = 1

BATCH_SIZE = config['batch_size']
IS_DIFFER = config['is_differ']
DEPTH = config['depth']

file_name = datetime.now().strftime('%Y%m%d_%H')
log_filename = f"log/train/{file_name}.log"

chpt_dir = f"{config['checkpoint_dir']}/{file_name}"
os.makedirs(chpt_dir, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_filename),
        logging.StreamHandler()
    ]
)

logging.info(f"CUDA is available: {torch.cuda.is_available()}")

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
    f"  Number of Epochs: {config['num_epochs']} "
    f"  Device: {config['device']} "
    f"  Checkpoint Directory: {config['checkpoint_dir']} "
    f"  Plot Directory: {config['plot_dir']} "
    f"  Log Interval: {config['log_interval']} "
    f"  Save Interval: {config['save_interval']}"
)

# 1. Data Loader
train_data = torch.load('dataset/train_dataset.pt')
train_dataset = DrugResponseDataset(
    gene_embeddings=train_data['gene_embeddings'],
    pathway_graphs=train_data['pathway_graphs'],
    substructure_embeddings=train_data['substructure_embeddings'],
    drug_graphs=train_data['drug_graphs'],
    labels=train_data['labels'],
    sample_indices=train_data['sample_indices'],
)

val_data = torch.load('dataset/val_dataset.pt')
val_dataset = DrugResponseDataset(
    gene_embeddings=val_data['gene_embeddings'],
    pathway_graphs=val_data['pathway_graphs'],
    substructure_embeddings=val_data['substructure_embeddings'],
    drug_graphs=val_data['drug_graphs'],
    labels=val_data['labels'],
    sample_indices=val_data['sample_indices'],
)

train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False, collate_fn=collate_fn)


# 2. Model Initialization
model = DrugResponseModel(NUM_PATHWAYS, NUM_GENES, NUM_SUBSTRUCTURES, GENE_EMBEDDING_DIM, SUBSTRUCTURE_EMBEDDING_DIM, HIDDEN_DIM, FINAL_DIM, OUTPUT_DIM, BATCH_SIZE, IS_DIFFER, DEPTH)
logging.info(f"Initial GPU memory usage (before moving model to device): "
             f"{torch.cuda.memory_allocated(config['device']) / 1e6:.2f} MB allocated, "
             f"{torch.cuda.memory_reserved(config['device']) / 1e6:.2f} MB reserved.")

model = model.to(config['device'])

# 모델 파라미터 개수 확인
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

logging.info(f"Total model parameters: {total_params:,}")
logging.info(f"Trainable model parameters: {trainable_params:,}")

# GPU 메모리 사용량 출력
logging.info(f"Initial GPU memory usage: {torch.cuda.memory_allocated(config['device']) / 1e6:.2f} MB allocated, "
             f"{torch.cuda.memory_reserved(config['device']) / 1e6:.2f} MB reserved.")

criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
logging.info(f"Model initialized on device: {config['device']}")


# 3. Helper Function
def process_batch(batch, model, criterion, device):
    gene_embeddings = batch['gene_embeddings'].to(device)
    pathway_graphs = batch['pathway_graphs'].to(device)
    substructure_embeddings = batch['substructure_embeddings'].to(device)
    drug_graphs = batch['drug_graphs'].to(device)
    labels = batch['labels'].to(device)

    with autocast(device_type="cuda" if config['device'].type == "cuda" else "cpu"):
        outputs = model(gene_embeddings, pathway_graphs, substructure_embeddings, drug_graphs)
        outputs = outputs.squeeze(dim=-1) 
        loss = criterion(outputs, labels) 

    preds = (torch.sigmoid(outputs) > 0.5).long() 
    correct_preds = (preds == labels).sum().item()
    total_samples = labels.size(0)

    return loss, correct_preds, total_samples

# 4. Training Loop
train_losses, val_losses = [], []
train_accuracies, val_accuracies = [], []

scaler = GradScaler()

for epoch in range(config['num_epochs']):
    epoch_start = time.time()
    logging.info(f"Epoch [{epoch+1}/{config['num_epochs']}] started.")

    # Training Phase
    model.train()
    total_train_loss, correct_train_preds, total_train_samples = 0, 0, 0

    for batch_idx, batch in enumerate(tqdm(train_loader, desc=f"Train Epoch {epoch+1}")):
        optimizer.zero_grad()
        
        with autocast(device_type="cuda" if config['device'].type == "cuda" else "cpu"):
            loss, correct_preds, total_samples = process_batch(batch, model, criterion, config['device'])

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_train_loss += loss.item()
        correct_train_preds += correct_preds
        total_train_samples += total_samples

        if batch_idx % config['log_interval'] == 0:
            logging.info(f"Batch {batch_idx+1}: Loss: {loss.item():.4f}, ")            

        clear_cache()

    train_loss = total_train_loss / len(train_loader)
    train_accuracy = correct_train_preds / total_train_samples
    train_losses.append(train_loss)
    train_accuracies.append(train_accuracy)

    # Validation Phase
    model.eval()
    total_val_loss, correct_val_preds, total_val_samples = 0, 0, 0

    with torch.no_grad():
        with autocast(device_type="cuda" if config['device'].type == "cuda" else "cpu"):
            for batch in tqdm(val_loader, desc=f"Validation Epoch {epoch+1}"):
                loss, correct_preds, total_samples = process_batch(batch, model, criterion, config['device'])
                total_val_loss += loss.item()
                correct_val_preds += correct_preds
                total_val_samples += total_samples

    val_loss = total_val_loss / len(val_loader)
    val_accuracy = correct_val_preds / total_val_samples
    val_losses.append(val_loss)
    val_accuracies.append(val_accuracy)

    logging.info(f"Epoch [{epoch+1}/{config['num_epochs']}] completed. "
                 f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, "
                 f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")

    # Save Checkpoint
    if (epoch + 1) % config['save_interval'] == 0:
        checkpoint_path = f"{chpt_dir}/ckpt_epoch_{epoch+1}.pth" # 체크 포인트 수정됨
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
            'train_accuracy': train_accuracy,
            'val_accuracy': val_accuracy
        }, checkpoint_path)
        logging.info(f"Model checkpoint saved: {checkpoint_path}")

    plot_statics(file_name, train_losses, val_losses, train_accuracies, val_accuracies) # 추가


plot_statics(file_name, train_losses, val_losses, train_accuracies, val_accuracies)
logging.info(f"Plots saved in directory Plots.")
