import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
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
    'batch_size': 4,
    'depth' : 2,
    'learning_rate': 0.0001,
    'num_epochs': 100,
    'early_stopping_patience': 5,
    'checkpoint_dir': './checkpoints',
    'plot_dir': './plots',
    'log_interval': 50,
    'save_interval': 100,
}

MAX_GENE_SLOTS = 218  # 유전자 최대 개수
MAX_DRUG_SUBSTRUCTURES = 17 # 하위구조 최대 개수
GENE_LAYER_EMBEDDING_DIM = 64
SUBSTRUCTURE_LAYER_EMBEDDING_DIM = 64
CROSS_ATTN_EMBEDDING_DIM = 64
FINAL_EMBEDDING_DIM = 128
OUTPUT_DIM = 1

BATCH_SIZE = config['batch_size']
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
    "Embedding Dimensions and Hidden Layers: "
    f"  GENE_EMBEDDING_DIM: {GENE_LAYER_EMBEDDING_DIM} "
    f"  SUBSTRUCTURE_EMBEDDING_DIM: {SUBSTRUCTURE_LAYER_EMBEDDING_DIM} "
    f"  FINAL_DIM: {FINAL_EMBEDDING_DIM} "
    f"  OUTPUT_DIM: {OUTPUT_DIM}")
logging.info(
    "Training Configuration: "
    f"  BATCH_SIZE: {BATCH_SIZE} "
    f"  Learning Rate: {config['learning_rate']} "
    f"  Number of Epochs: {config['num_epochs']} "
)
logging.info(f"CUDA is available: {torch.cuda.is_available()}")

# Seed Setting
seed = 42
set_seed(42)

# Common Data Load
pathway_masks = torch.load('./input/pathway_mask.pt')
pathway_laplacian_embeddings = torch.load('./input/pathway_laplacian_embeddings_4.pt') 

# Helper Function
def process_batch(batch_idx, batch, epoch, model, criterion, device, mode="train"):
    gene_embeddings = batch['gene_embeddings'].to(device) # [Batch, Pathway_num, Max_Gene]
    drug_embeddings = batch['drug_embeddings'].to(device) # [Batch, Max_Sub]
    drug_spectral_embeddings = batch['drug_spectral_embeddings'].to(device) # [Batch, Max_Sub]
    drug_masks = batch['drug_masks'].to(device) # [Batch, Max_Sub]
    labels = batch['labels'].to(device) # [Batch]
    sample_indices = batch['sample_indices']

    outputs, gene2sub_weights, sub2gene_weights, final_pathway_embedding, final_drug_embedding = model(gene_embeddings, drug_embeddings, drug_spectral_embeddings, drug_masks)    
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
        torch.save(sample_indices, f"{save_dir}/B{batch_idx}_samples.pt") # (세포주, 약물)

    return outputs, loss, rmse, sample_indices, labels

# 5-Fold Cross-Validation
best_val_rmse_overall = float('inf')
best_checkpoint_overall = None
best_fold_id = None

for fold in range(1, 6):
    logging.info(f"🚩 Starting Fold {fold}")

    # Fold Directory Setting
    fold_checkpoint_dir = f"{chpt_dir}/fold_{fold}"
    fold_result_dir = f"results_cv/{FILE_NAME}/fold_{fold}"
    attn_dir = f"weights/{FILE_NAME}/fold_{fold}"
    os.makedirs(fold_checkpoint_dir, exist_ok=True)
    os.makedirs(fold_result_dir, exist_ok=True)
    os.makedirs(attn_dir, exist_ok=True)
    
    # Fold Data Load & Init Model
    fold_data = torch.load(f'../0_dataset_CV_laplacian/cross_valid_fold_{fold}.pt')
    train_dataset = DrugResponseDataset(**fold_data['train'])
    val_dataset = DrugResponseDataset(**fold_data['validation'])
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
    
    model = DrugResponseModel(
        pathway_masks=pathway_masks,
        pathway_laplacian_embeddings=pathway_laplacian_embeddings, 
        gene_layer_dim=GENE_LAYER_EMBEDDING_DIM,
        substructure_layer_dim=SUBSTRUCTURE_LAYER_EMBEDDING_DIM,
        cross_attn_dim=CROSS_ATTN_EMBEDDING_DIM,
        final_dim=FINAL_EMBEDDING_DIM,
        max_gene_slots=MAX_GENE_SLOTS, 
        max_drug_substructures=MAX_DRUG_SUBSTRUCTURES 
    ).to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
    
    # Training Loop
    train_losses, val_losses = [], []
    train_rmses, val_rmses = [], []

    best_val_rmse = float('inf')
    patience_counter = 0

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
            outputs, loss, rmse, sample_indices, labels  = process_batch(batch_idx, batch, epoch+1, model, criterion, config['device'], mode="train")
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()
            total_train_rmse += rmse

            if batch_idx % config['log_interval'] == 0:
                    logging.info(f"Batch {batch_idx+1}: Loss: {loss.item():.4f}, ")   

            train_actuals.extend(labels) # Labels
            train_predictions.extend(outputs) # Predictions
            train_samples.extend(sample_indices) # Samples

        train_loss = total_train_loss / len(train_loader)
        train_rmse = total_train_rmse / len(train_loader) 
        train_losses.append(train_loss)
        train_rmses.append(train_rmse)

        # Validation Phase
        model.eval()
        total_val_loss, total_val_rmse = 0, 0

        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Validation Epoch {epoch+1}"):
                outputs, loss, rmse, sample_indices, labels = process_batch(batch_idx, batch, epoch+1, model, criterion, config['device'], mode="val")
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

        logging.info(f"Fold {fold} Epoch [{epoch+1}/{config['num_epochs']}] completed. \n"
                     f"Train Loss: {train_loss:.4f}, Train RMSE: {train_rmse:.4f}, Train PCC: {train_pcc:.4f}, Train SCC: {train_scc:.4f},\n"
                     f"Val Loss: {val_loss:.4f}, Val RMSE: {val_rmse:.4f} Val PCC: {val_pcc:.4f}, Val SCC: {val_scc:.4f} \n"
        )

        # Check Early Stopping
        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            patience_counter = 0

            # Save Best Model & Results
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
            }, os.path.join(fold_checkpoint_dir, "best_model.pth"))

            torch.save({
                "actuals": train_actuals,
                "predictions": train_predictions,
                "drug_labels": train_samples
            }, os.path.join(fold_result_dir, f"train_results_epoch_{epoch+1}.pt"))

            torch.save({
                "actuals": val_actuals,
                "predictions": val_predictions,
                "drug_labels": val_samples
            }, os.path.join(fold_result_dir, f"val_results_epoch_{epoch+1}.pt"))

            logging.info(f"✅ New best model and results saved at epoch {epoch+1} (RMSE: {val_rmse:.4f} in directory {fold_result_dir})")

            # Update Overall Best Model
            if val_rmse < best_val_rmse_overall:
                best_val_rmse_overall = val_rmse
                best_checkpoint_overall = os.path.join(chpt_dir, "overall_best_model.pth")
                best_fold_id = fold

        else:
            patience_counter += 1
            logging.info(f"⚠️ No improvement in validation RMSE. Patience: {patience_counter}/{config['early_stopping_patience']}")
            if patience_counter >= config['early_stopping_patience']:
                logging.info("🛑 Early stopping triggered. Training terminated.")
                break

        plot_statics( FILE_NAME, f"Fold {fold}", train_losses, val_losses, train_rmses, val_rmses)

# Cross Validation Best Results
logging.info("🏁 Cross Validation Completed.")
logging.info(f"Best Fold: {best_fold_id}")
logging.info(f"Best RMSE: {best_val_rmse_overall:.4f}")
logging.info(f"Best Checkpoint Path: {best_checkpoint_overall}")


# Save Loss Plots
plot_statics(FILE_NAME, "Total", train_losses, val_losses, train_rmses, val_rmses)
