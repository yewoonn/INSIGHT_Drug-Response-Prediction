import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import logging
from tqdm import tqdm
from datetime import datetime
from scipy.stats import spearmanr, pearsonr
import numpy as np

from dataset import DrugResponseDataset, collate_fn
from model import DrugResponseModel
from utils import plot_statics, set_seed

# 기본 하이퍼파라미터
config = {
    'device': torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
    'batch_size': 32,
    'depth': 2,
    'learning_rate': 0.0001,
    'num_epochs': 50,
    'checkpoint_dir': './checkpoints',
    'plot_dir': './plots',
    'log_interval': 50,
    'save_interval': 100,
}

NUM_PATHWAYS = 314
NUM_MAX_GENES = 264
NUM_MAX_SUBSTRUCTURES = 17
GENE_LAYER_EMBEDDING_DIM = 128
SUBSTRUCTURE_LAYER_EMBEDDING_DIM = 768
CROSS_ATTN_EMBEDDING_DIM = 128
PATHWAY_GRAPH_EMBEDDING_DIM = 256
DRUG_GRAPH_EMBEDDING_DIM = 128
FINAL_EMBEDDING_DIM = 512
OUTPUT_DIM = 1

DEVICE = config['device']
BATCH_SIZE = config['batch_size']
FILE_NAME = datetime.now().strftime('%Y%m%d_%H')
SAVE_INTERVALS = config['save_interval']

# Logging 설정
log_filename = f"log/train/{FILE_NAME}.log"
os.makedirs(os.path.dirname(log_filename), exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_filename),
        logging.StreamHandler()
    ]
)

# Seed 설정
set_seed(42)

# 공통 데이터 로드
pathway_graphs = torch.load('./input_0307/pathway_graphs_128.pt')
pathway_masks = torch.load('./input_0307/pathway_mask.pt')

# Helper function
def process_batch(batch_idx, batch, epoch, model, criterion, device, mode="train"):
    gene_embeddings = batch['gene_embeddings'].to(device)
    drug_embeddings = batch['drug_embeddings'].to(device)
    drug_graphs = batch['drug_graphs'].to(device)
    drug_masks = batch['drug_masks'].to(device)
    labels = batch['labels'].to(device)
    sample_indices = batch['sample_indices']

    outputs, gene2sub_weights, sub2gene_weights, final_pathway_embedding, final_drug_embedding = model(
        gene_embeddings, drug_embeddings, drug_graphs, drug_masks)
    outputs = outputs.squeeze(dim=-1)
    loss = criterion(outputs, labels)
    rmse = torch.sqrt(loss).item()

    if mode == "train" and (epoch % SAVE_INTERVALS == 0):
        save_dir = f"{attn_dir}/epoch_{epoch}"
        os.makedirs(save_dir, exist_ok=True)
        torch.save(gene2sub_weights.detach().cpu(), f"{save_dir}/B{batch_idx}_gene2sub_weight.pt")
        torch.save(sub2gene_weights.detach().cpu(), f"{save_dir}/B{batch_idx}_sub2gene_weight.pt")
        torch.save(final_pathway_embedding.detach().cpu(), f"{save_dir}/B{batch_idx}_pathway_embedding.pt")
        torch.save(final_drug_embedding.detach().cpu(), f"{save_dir}/B{batch_idx}_drug_embedding.pt")
        torch.save(sample_indices, f"{save_dir}/B{batch_idx}_samples.pt")

    return outputs, loss, rmse, sample_indices, labels

# 5-Fold Cross-Validation
best_val_rmse_overall = float('inf')
best_checkpoint_overall = None
best_fold_id = None

for fold in range(1, 6):
    logging.info(f"🚩 Starting Fold {fold}")

    # 폴드별 디렉토리
    fold_checkpoint_dir = f"{config['checkpoint_dir']}/fold_{fold}"
    fold_result_dir = f"results_cv/{FILE_NAME}/fold_{fold}"
    attn_dir = f"weights/{FILE_NAME}/fold_{fold}"
    os.makedirs(fold_checkpoint_dir, exist_ok=True)
    os.makedirs(fold_result_dir, exist_ok=True)
    os.makedirs(attn_dir, exist_ok=True)

    # 데이터 로드
    fold_data = torch.load(f'./dataset_full_CV/cross_valid_fold_{fold}.pt')
    train_dataset = DrugResponseDataset(**fold_data['train'])
    val_dataset = DrugResponseDataset(**fold_data['validation'])
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

    # 모델 초기화
    model = DrugResponseModel(
        pathway_graphs=pathway_graphs,
        pathway_masks=pathway_masks,
        num_pathways=NUM_PATHWAYS,
        num_genes=NUM_MAX_GENES,
        num_substructures=NUM_MAX_SUBSTRUCTURES,
        gene_layer_dim=GENE_LAYER_EMBEDDING_DIM,
        substructure_layer_dim=SUBSTRUCTURE_LAYER_EMBEDDING_DIM,
        cross_attn_dim=CROSS_ATTN_EMBEDDING_DIM,
        pathway_graph_dim=PATHWAY_GRAPH_EMBEDDING_DIM,
        drug_graph_dim=DRUG_GRAPH_EMBEDDING_DIM,
        final_dim=FINAL_EMBEDDING_DIM,
        output_dim=OUTPUT_DIM,
        batch_size=BATCH_SIZE,
        depth=config['depth'],
        save_intervals=SAVE_INTERVALS,
        file_name=FILE_NAME,
        device=DEVICE
    ).to(DEVICE)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])

    train_losses, val_losses, train_rmses, val_rmses = [], [], [], []

    best_val_rmse = float('inf')
    best_model_state = None
    best_epoch = -1

    for epoch in range(config['num_epochs']):
        model.train()
        total_train_loss, total_train_rmse = 0.0, 0.0
        train_actuals, train_predictions, train_samples = [], [], []

        for batch_idx, batch in enumerate(tqdm(train_loader, desc=f"Fold {fold} Train Epoch {epoch+1}")):
            optimizer.zero_grad()
            outputs, loss, rmse, sample_indices, labels = process_batch(batch_idx, batch, epoch+1, model, criterion, DEVICE, mode="train")
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()
            total_train_rmse += rmse
            train_actuals.extend(labels)
            train_predictions.extend(outputs)
            train_samples.extend(sample_indices)

        train_loss = total_train_loss / len(train_loader)
        train_rmse = total_train_rmse / len(train_loader)
        train_losses.append(train_loss)
        train_rmses.append(train_rmse)

        model.eval()
        total_val_loss, total_val_rmse = 0.0, 0.0
        val_actuals, val_predictions, val_samples = [], [], []

        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(val_loader, desc=f"Fold {fold} Val Epoch {epoch+1}")):
                outputs, loss, rmse, sample_indices, labels = process_batch(batch_idx, batch, epoch+1, model, criterion, DEVICE, mode="val")
                total_val_loss += loss.item()
                total_val_rmse += rmse
                val_actuals.extend(labels)
                val_predictions.extend(outputs)
                val_samples.extend(sample_indices)

        val_loss = total_val_loss / len(val_loader)
        val_rmse = total_val_rmse / len(val_loader)
        val_losses.append(val_loss)
        val_rmses.append(val_rmse)

        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            best_epoch = epoch + 1
            best_model_state = {
                'epoch': best_epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
               'train_loss': train_loss,
                'val_loss': val_loss,
                'val_rmse': val_rmse,
            }

        # 상관계수 계산
        train_actuals_np = torch.stack(train_actuals).cpu().numpy()
        train_predictions_np = torch.stack(train_predictions).detach().cpu().numpy()
        val_actuals_np = torch.stack(val_actuals).cpu().numpy()
        val_predictions_np = torch.stack(val_predictions).detach().cpu().numpy()

        train_pcc = pearsonr(train_actuals_np, train_predictions_np)[0]
        train_scc = spearmanr(train_actuals_np, train_predictions_np)[0]
        val_pcc = pearsonr(val_actuals_np, val_predictions_np)[0]
        val_scc = spearmanr(val_actuals_np, val_predictions_np)[0]

        logging.info(f"Fold {fold} Epoch [{epoch+1}/{config['num_epochs']}] completed. \n"
                     f"Train Loss: {train_loss:.4f}, Train RMSE: {train_rmse:.4f}, PCC: {train_pcc:.4f}, SCC: {train_scc:.4f},\n"
                     f"Val Loss: {val_loss:.4f}, Val RMSE: {val_rmse:.4f}, PCC: {val_pcc:.4f}, SCC: {val_scc:.4f}")

        # 결과 저장
        torch.save({"actuals": train_actuals, "predictions": train_predictions, "drug_labels": train_samples},
                   os.path.join(fold_result_dir, f"train_results_epoch_{epoch+1}.pt"))
        torch.save({"actuals": val_actuals, "predictions": val_predictions, "drug_labels": val_samples},
                   os.path.join(fold_result_dir, f"val_results_epoch_{epoch+1}.pt"))

    # Fold 별 Best 체크포인트 저장
    torch.save(best_model_state, f"{fold_checkpoint_dir}/best_checkpoint.pth")

    if best_val_rmse < best_val_rmse_overall:
        best_val_rmse_overall = best_val_rmse
        best_checkpoint_overall = best_model_state
        best_fold_id = fold

    # 그래프 저장
    plot_statics(f"{fold_result_dir}/fold_{fold}", train_losses, val_losses, train_rmses, val_rmses)
    logging.info(f"Fold {fold} completed. Average Val Loss: {np.mean(val_losses):.4f}, RMSE: {np.mean(val_rmses):.4f}")


# 모든 Fold 중 Best 체크포인트 저장
torch.save(best_checkpoint_overall, f"{config['checkpoint_dir']}/best_model_overall.pth")
logging.info(f"Best model found at Fold {best_fold_id} with RMSE {best_val_rmse_overall:.4f} saved.")

logging.info("5-fold cross-validation 완료.")
