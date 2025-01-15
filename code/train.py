import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.amp import GradScaler, autocast
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

import logging
import time
from tqdm import tqdm
from datetime import datetime
import os

from dataset import DrugResponseDataset, collate_fn
from model import DrugResponseModel
from utils import plot_statics, AttentionLogger

os.environ['TZ'] = 'Asia/Seoul'
time.tzset()  # Unix 환경에서 적용
# os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:1024' # OR 512
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

# Configuration
config = {
    'device': torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
    'batch_size': 64,
    'is_differ' : True,
    'depth' : 2,
    'learning_rate': 0.0005,
    'weight_decay': 0.001,
    'num_epochs': 10,
    'checkpoint_dir': './checkpoints', # ckpt 디렉토리
    'plot_dir': './plots',
    'log_interval': 10, # batch 별 log 출력 간격
    'save_interval': 10, # ckpt 저장할 epoch 간격
    'save_pathways' : [0, 79]
}

NUM_CELL_LINES = 1280
NUM_PATHWAYS = 312
NUM_GENES = 3848
NUM_DRUGS = 10
NUM_SUBSTRUCTURES = 194

GENE_EMBEDDING_DIM = 32
SUBSTRUCTURE_EMBEDDING_DIM = 32
HIDDEN_DIM = 32
FINAL_DIM = 16
OUTPUT_DIM = 1

BATCH_SIZE = config['batch_size']
IS_DIFFER = config['is_differ']
DEPTH = config['depth']

FILE_NAME = datetime.now().strftime('%Y%m%d_%H')
SAVE_INTERVALS = config['save_interval']
SAVE_PATHWAYS = config['save_pathways']
log_filename = f"log/train/{FILE_NAME}.log"

chpt_dir = f"{config['checkpoint_dir']}/{FILE_NAME}"
os.makedirs(chpt_dir, exist_ok=True)

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
train_data = torch.load('dataset/train_dataset.pt')
train_dataset = DrugResponseDataset(
    gene_embeddings=train_data['gene_embeddings'],
    drug_embeddings=train_data['drug_embeddings'],
    drug_graphs=train_data['drug_graphs'],
    labels=train_data['labels'],
    sample_indices=train_data['sample_indices'],
)

val_data = torch.load('dataset/val_dataset.pt')
val_dataset = DrugResponseDataset(
    gene_embeddings=val_data['gene_embeddings'],
    drug_embeddings=val_data['drug_embeddings'],
    drug_graphs=val_data['drug_graphs'],
    labels=val_data['labels'],
    sample_indices=val_data['sample_indices'],
)

train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False, collate_fn=collate_fn)

pathway_genes_dict = torch.load('./input/0_pathway_genes_dict.pt')
pathway_graphs = torch.load('./input/0_pathway_graph.pt')

# 2. Model Initialization
attn_logger = AttentionLogger()
model = DrugResponseModel(NUM_PATHWAYS, pathway_graphs, pathway_genes_dict, NUM_GENES, NUM_SUBSTRUCTURES, GENE_EMBEDDING_DIM, SUBSTRUCTURE_EMBEDDING_DIM, HIDDEN_DIM, FINAL_DIM, OUTPUT_DIM, BATCH_SIZE, IS_DIFFER, DEPTH, SAVE_INTERVALS, SAVE_PATHWAYS,FILE_NAME, attn_logger)
logging.info(f"Initial GPU memory usage (before moving model to device): "
             f"{torch.cuda.memory_allocated(config['device']) / 1e6:.2f} MB allocated, "
             f"{torch.cuda.memory_reserved(config['device']) / 1e6:.2f} MB reserved.")

model = model.to(config['device'])

# 모델 파라미터 개수 확인
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

logging.info(f"Total model parameters: {total_params:,}")
logging.info(f"Trainable model parameters: {trainable_params:,}")

criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
logging.info(f"Model initialized on device: {config['device']}")


# 3. Helper Function
def process_batch(batch, epoch, model, criterion, device):
    gene_embeddings = batch['gene_embeddings'].to(device)
    drug_embeddings = batch['drug_embeddings'].to(device)
    drug_graphs = batch['drug_graphs'].to(device)
    labels = batch['labels'].to(device)
    sample_indices = batch['sample_indices'] 
    
    with autocast(device_type="cuda" if config['device'].type == "cuda" else "cpu"):
        outputs = model(gene_embeddings, drug_embeddings, drug_graphs, epoch, sample_indices)
        outputs = outputs.squeeze(dim=-1) 
        loss = criterion(outputs, labels) 

    preds = (torch.sigmoid(outputs) > 0.5).long() 
    correct_preds = (preds == labels).sum().item()
    total_samples = labels.size(0)
    incorrect_indices = [idx for idx, (pred, label) in enumerate(zip(preds, labels)) if pred != label]

    return loss, correct_preds, total_samples, outputs, incorrect_indices

# 4. Training Loop
train_losses, val_losses = [], []
train_accuracies, val_accuracies = [], []
train_precisions, val_precisions = [], []
train_recalls, val_recalls = [], []
train_f1s, val_f1s = [], []

scaler = GradScaler()

for epoch in range(config['num_epochs']):
    epoch_start = time.time()
    logging.info(f"Epoch [{epoch+1}/{config['num_epochs']}] started.")

    # Training Phase
    model.train()
    total_train_loss, correct_train_preds, total_train_samples = 0, 0, 0
    y_true_train, y_pred_train = [], []
    incorrect_train_samples = [] 

    for batch_idx, batch in enumerate(tqdm(train_loader, desc=f"Train Epoch {epoch+1}")):
        optimizer.zero_grad()
        
        with autocast(device_type="cuda" if config['device'].type == "cuda" else "cpu"):
            loss, correct_preds, total_samples, outputs, incorrect_indices= process_batch(batch, epoch+1, model, criterion, config['device'])

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        incorrect_train_samples.extend([batch['sample_indices'][i] for i in incorrect_indices])

        total_train_loss += loss.item()
        correct_train_preds += correct_preds
        total_train_samples += total_samples

        y_true_train.extend(batch['labels'].cpu().numpy())  
        y_pred_train.extend((torch.sigmoid(outputs).detach().cpu().numpy() > 0.5).astype(int))


        if batch_idx % config['log_interval'] == 0:
            logging.info(f"Batch {batch_idx+1}: Loss: {loss.item():.4f}, ")            

        if ((epoch+1) % SAVE_INTERVALS == 0):
            model.attn_logger.flush_to_json(model.file_name, epoch+1)

    train_loss = total_train_loss / len(train_loader)
    train_accuracy = accuracy_score(y_true_train, y_pred_train)
    train_precision = precision_score(y_true_train, y_pred_train)
    train_recall = recall_score(y_true_train, y_pred_train)
    train_f1 = f1_score(y_true_train, y_pred_train)

    train_losses.append(train_loss)
    train_accuracies.append(train_accuracy)
    train_precisions.append(train_precision)
    train_recalls.append(train_recall)
    train_f1s.append(train_f1)

    # Validation Phase
    model.eval()
    total_val_loss, correct_val_preds, total_val_samples = 0, 0, 0
    y_true_val, y_pred_val = [], []
    incorrect_val_samples = []

    with torch.no_grad():
        with autocast(device_type="cuda" if config['device'].type == "cuda" else "cpu"):
            for batch in tqdm(val_loader, desc=f"Validation Epoch {epoch+1}"):
                loss, correct_preds, total_samples, outputs, incorrect_indices = process_batch(batch, epoch+1, model, criterion, config['device'])
                incorrect_val_samples.extend([batch['sample_indices'][i] for i in incorrect_indices])

                total_val_loss += loss.item()
                correct_val_preds += correct_preds
                total_val_samples += total_samples

                y_true_val.extend(batch['labels'].cpu().numpy())  # GPU -> CPU 변환
                y_pred_val.extend((torch.sigmoid(outputs).detach().cpu().numpy() > 0.5).astype(int))  # GPU -> CPU 변환 후 NumPy 변환
            
    val_loss = total_val_loss / len(val_loader)
    val_accuracy = accuracy_score(y_true_val, y_pred_val)
    val_precision = precision_score(y_true_val, y_pred_val)
    val_recall = recall_score(y_true_val, y_pred_val)
    val_f1 = f1_score(y_true_val, y_pred_val)

    val_losses.append(val_loss)
    val_accuracies.append(val_accuracy)
    val_precisions.append(val_precision)
    val_recalls.append(val_recall)
    val_f1s.append(val_f1)

    logging.info(f"Epoch [{epoch+1}/{config['num_epochs']}] completed. "
                     f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, "
                     f"Train Precision: {train_precision:.4f}, Train Recall: {train_recall:.4f}, Train F1: {train_f1:.4f}, "
                     f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}, "
                     f"Val Precision: {val_precision:.4f}, Val Recall: {val_recall:.4f}, Val F1: {val_f1:.4f}")

    logging.info(f"==============Incorrect Samples in Epoch {epoch+1}=============="
        f"Incorrect Train Samples : {incorrect_train_samples}"
        f"Incorrect Validation Samples : {incorrect_val_samples}"
    )

    # Save Checkpoint
    if (epoch + 1) % SAVE_INTERVALS == 0:
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

    plot_statics(FILE_NAME, train_losses, val_losses, train_accuracies, val_accuracies) 
    

plot_statics(FILE_NAME, train_losses, val_losses, train_accuracies, val_accuracies)
logging.info(f"Plots saved in directory Plots.")
