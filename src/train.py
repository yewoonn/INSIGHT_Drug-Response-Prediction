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
import yaml
import argparse
from dataset import DrugResponseDataset, collate_fn
from model import DrugResponseModel
from utils import plot_statics, set_seed
import shutil

# Parse command line
def parse_args():
    parser = argparse.ArgumentParser(description='Drug Response Prediction Training')
    parser.add_argument('--config', type=str, default='config.yml', 
                       help='Path to configuration file (default: config.yml)')
    return parser.parse_args()

# Load configuration
def load_config(config_path='config.yml'):
    with open(config_path, 'r', encoding='utf-8') as file:
        config = yaml.safe_load(file)
    return config

args = parse_args()
config = load_config(args.config)

os.environ['TZ'] = config['system']['timezone']
time.tzset()
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = config['system']['pytorch_cuda_alloc_conf']

# Fixed directory paths
CHECKPOINT_DIR = "./checkpoints"
PLOT_DIR = "./plots"
LOG_DIR = "log/train"
WEIGHTS_DIR = "weights"
PREDICTION_DIR = "predictions"

device = config['training']['device']
if device == "cuda:0" and not torch.cuda.is_available():
    device = "cpu"
config['training']['device'] = torch.device(device)

# Extract parameters
training_config = config['training']
save_config = config['save']
model_config = config['model']
data_config = config['data']
system_config = config['system']

# --- 모델 파라미터 추출 수정 ---
GENE_FFN_OUTPUT_DIM = model_config['gene_ffn_output_dim']
DRUG_FFN_OUTPUT_DIM = model_config['drug_ffn_output_dim'] # ChemBERTa FFN 출력 차원
MAX_GENE_SLOTS = model_config['max_gene_slots']
GENE_INPUT_DIM = model_config['gene_input_dim'] # Gene FFN 입력 차원
DRUG_INPUT_DIM = model_config.get('drug_input_dim', 768) # ChemBERTa 입력 차원
IS_DIFFER = model_config.get('isDiffer', True) # 차등 크로스 어텐션 사용 여부
CROSS_ATTN_EMBEDDING_DIM = model_config['cross_attn_embedding_dim']
FINAL_EMBEDDING_DIM = model_config['final_embedding_dim']
OUTPUT_DIM = model_config['output_dim']

BATCH_SIZE = training_config['batch_size']
FILE_NAME = datetime.now().strftime('%Y%m%d_%H')
DEVICE = training_config['device']
SAVE_FOLD_NUMBER = save_config['save_fold_number'] 

log_filename = f"{LOG_DIR}/{FILE_NAME}.log"
chpt_dir = f"{CHECKPOINT_DIR}/{FILE_NAME}"
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
    "Embedding Dimensions and Hidden Layers: "
    f"  GENE_FFN_OUTPUT_DIM: {GENE_FFN_OUTPUT_DIM} "
    f"  DRUG_FFN_OUTPUT_DIM: {DRUG_FFN_OUTPUT_DIM} "
    f"  DRUG_INPUT_DIM: {DRUG_INPUT_DIM} "
    f"  FINAL_DIM: {FINAL_EMBEDDING_DIM} "
    f"  OUTPUT_DIM: {OUTPUT_DIM}")
logging.info(
    "Training Configuration: "
    f"  BATCH_SIZE: {BATCH_SIZE} "
    f"  Learning Rate: {training_config['learning_rate']} "
    f"  Number of Epochs: {training_config['num_epochs']} "
)
logging.info(f"CUDA is available: {torch.cuda.is_available()}")

# Seed Setting
set_seed(system_config['seed'])

# Common Data Load
# Removed unused pathway_masks and pathway_laplacian_embeddings loading
gene_10fb_embeddings = torch.load(data_config['gene_10fb_embeddings_path'])
pathway_gene_indices = torch.load(data_config['pathway_gene_indices_path'])

# Helper Function
# train.py의 process_batch 함수
def process_batch(batch_idx, batch, epoch, model, criterion, device, mode="train", is_best_epoch=False, save_weights=False, save_dir_root=None):
    gene_embeddings = batch['gene_embeddings'].to(device)
    drug_chembert_embeddings = batch['drug_chembert_embeddings'].to(device)  # ChemBERTa embeddings
    labels = batch['labels'].to(device)
    sample_indices = batch['sample_indices']

    # --- 새로운 모델 반환값 ---
    outputs, path2drug_weights, drug2path_weights, gene_embedded_value, drug_embeddings, pathway_embeddings = model(gene_embeddings, drug_chembert_embeddings)
    outputs = outputs.squeeze(dim=-1) 
    loss = criterion(outputs, labels) 

    rmse = torch.sqrt(loss).item()  

    # --- Save weights 부분 수정 ---
    if mode == "train" and save_weights and save_config['isSave']:
        save_dir = f"{save_dir_root}/current_epoch"
        os.makedirs(save_dir, exist_ok=True)
        torch.save(path2drug_weights.detach().cpu(), f"{save_dir}/B{batch_idx}_path2drug_weight.pt")
        torch.save(drug2path_weights.detach().cpu(), f"{save_dir}/B{batch_idx}_drug2path_weight.pt")
        
        torch.save(gene_embedded_value.detach().cpu(), f"{save_dir}/B{batch_idx}_pathway_value_embed.pt")
        torch.save(drug_embeddings.detach().cpu(), f"{save_dir}/B{batch_idx}_drug_embed.pt")
        torch.save(pathway_embeddings.detach().cpu(), f"{save_dir}/B{batch_idx}_pathway_embed.pt")
        torch.save(sample_indices, f"{save_dir}/B{batch_idx}_samples.pt")

    return outputs, loss, rmse, sample_indices, labels

# 5-Fold Cross-Validation
for fold in range(1, 6):
    logging.info(f"🚩 Starting Fold {fold}")

    # Fold Directory Setting
    fold_checkpoint_dir = f"{chpt_dir}/fold_{fold}"
    fold_result_dir = f"{PREDICTION_DIR}/{FILE_NAME}/fold_{fold}"
    fold_attn_dir = f"{WEIGHTS_DIR}/{FILE_NAME}/fold_{fold}"
    os.makedirs(fold_checkpoint_dir, exist_ok=True)
    os.makedirs(fold_result_dir, exist_ok=True)
    os.makedirs(fold_attn_dir, exist_ok=True)
    
    # Fold Data Load & Init Model
    fold_data = torch.load(f'{data_config["cross_validation_data_dir"]}/cross_valid_fold_{fold}.pt')
    
    # gene_embeddings를 10fb 데이터로 교체
    fold_data['train']['gene_embeddings'] = gene_10fb_embeddings
    fold_data['validation']['gene_embeddings'] = gene_10fb_embeddings
    
    train_dataset = DrugResponseDataset(**fold_data['train'])
    val_dataset = DrugResponseDataset(**fold_data['validation'])
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn, num_workers=4, pin_memory=True)
    
        
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
        final_dim_reduction_factor=model_config['final_dim_reduction_factor']
    ).to(DEVICE)
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(
        model.parameters(), 
        lr=training_config['learning_rate'], 
        weight_decay=training_config.get('weight_decay', 0) # <-- weight_decay 추가
    )
    
    # Training Loop
    train_rmses, val_rmses = [], []

    best_val_rmse = float('inf')
    patience_counter = 0

    for epoch in range(training_config['num_epochs']):
        epoch_start = time.time()
        logging.info(f"Epoch [{epoch+1}/{training_config['num_epochs']}] started.")

        train_actuals, train_predictions, train_samples = [], [], []
        val_actuals, val_predictions, val_samples = [], [], []

        # Training Phase
        model.train()
        total_train_se, total_train_samples = 0, 0

        for batch_idx, batch in enumerate(tqdm(train_loader, desc=f"Train Epoch {epoch+1}")):
            optimizer.zero_grad()
            save_weights_this_epoch = (fold == SAVE_FOLD_NUMBER) and (epoch + 1 >= 40) and save_config['isSave'] # After 40 Epochs, Specific Fold for File Memorization Save
            outputs, loss, rmse, sample_indices, labels  = process_batch(batch_idx, batch, epoch+1, model, criterion, DEVICE, mode="train", is_best_epoch=False, save_weights=save_weights_this_epoch, save_dir_root=fold_attn_dir)
            loss.backward()
            optimizer.step()

            se = ((outputs.detach() - labels.detach()) ** 2).sum()
            total_train_se += se.item()
            total_train_samples += labels.numel()

            train_actuals.extend(labels.detach().cpu()) # Labels
            train_predictions.extend(outputs.detach().cpu()) # Predictions
            train_samples.extend(sample_indices) # Samples

        train_rmse = (total_train_se / total_train_samples) ** 0.5         
        train_rmses.append(train_rmse)

        # Validation Phase
        model.eval()
        total_val_se, total_val_samples = 0, 0

        with torch.no_grad():
            for val_idx, batch in enumerate(tqdm(val_loader, desc=f"Validation Epoch {epoch+1}")):
                outputs, loss, rmse, sample_indices, labels = process_batch(val_idx, batch, epoch+1, model, criterion, DEVICE, mode="val", is_best_epoch=False, save_weights=False)
                se = ((outputs.detach() - labels.detach()) ** 2).sum()
                total_val_se += se.item()
                total_val_samples += labels.numel()

                val_actuals.extend(labels.detach().cpu())
                val_predictions.extend(outputs.detach().cpu())
                val_samples.extend(sample_indices)

        val_rmse = (total_val_se / total_val_samples) ** 0.5         
        val_rmses.append(val_rmse)

        # SCC/PCC
        train_actuals_np = torch.stack(train_actuals).cpu().numpy()
        train_predictions_np = torch.stack(train_predictions).detach().cpu().numpy()
        val_actuals_np = torch.stack(val_actuals).cpu().numpy()
        val_predictions_np = torch.stack(val_predictions).detach().cpu().numpy()

        train_pcc = pearsonr(train_actuals_np, train_predictions_np)[0]
        train_scc = spearmanr(train_actuals_np, train_predictions_np)[0]
        val_pcc = pearsonr(val_actuals_np, val_predictions_np)[0]
        val_scc = spearmanr(val_actuals_np, val_predictions_np)[0]

        logging.info(f"Fold {fold} Epoch [{epoch+1}/{training_config['num_epochs']}] completed. \n"
                     f"Train RMSE: {train_rmse:.4f}, Train PCC: {train_pcc:.4f}, Train SCC: {train_scc:.4f}\n"
                     f"Val RMSE: {val_rmse:.4f}, Val PCC: {val_pcc:.4f}, Val SCC: {val_scc:.4f} \n"
        )

        # Check Early Stopping and manage weights
        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            patience_counter = 0

            # Save Best Model
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_rmse': train_rmse,
                'val_rmse': val_rmse,
            }, os.path.join(fold_checkpoint_dir, "best_model.pth"))

            # Current epoch weights → best epoch weights
            current_weights_dir = f"{fold_attn_dir}/current_epoch"
            best_weights_dir = f"{fold_attn_dir}/best_epoch"
            
            # Remove old best weights
            if os.path.exists(best_weights_dir):
                import shutil
                shutil.rmtree(best_weights_dir)
            
            # Move current weights → best weights
            if os.path.exists(current_weights_dir):
                os.rename(current_weights_dir, best_weights_dir)
                logging.info(f"✅ Weights saved for Fold {fold}, Epoch {epoch+1} (RMSE: {val_rmse:.4f})")

        else:
            patience_counter += 1
            
            # Delete current epoch weights
            current_weights_dir = f"{fold_attn_dir}/current_epoch"
            if os.path.exists(current_weights_dir):
                import shutil
                shutil.rmtree(current_weights_dir)
                logging.info(f"🗑️ Weights deleted for Fold {fold}, Epoch {epoch+1} (not best)")
            
            logging.info(f"⚠️ No improvement in validation RMSE. Patience: {patience_counter}/{training_config['early_stopping_patience']}")
            if patience_counter >= training_config['early_stopping_patience']:
                logging.info("🛑 Early stopping triggered. Training terminated.")
                break

        plot_statics( FILE_NAME, f"Fold {fold}", train_rmses, val_rmses)

    if fold != SAVE_FOLD_NUMBER:  # Only keep the specified fold
        current_fold_weights_dir = f"{WEIGHTS_DIR}/{FILE_NAME}/fold_{fold}"
        current_fold_results_dir = f"{PREDICTION_DIR}/{FILE_NAME}/fold_{fold}"
        
        if os.path.exists(current_fold_weights_dir):
            shutil.rmtree(current_fold_weights_dir)
            logging.info(f"🗑️ Deleted weights for Fold {fold} (not Fold {SAVE_FOLD_NUMBER})")
        
        if os.path.exists(current_fold_results_dir):
            shutil.rmtree(current_fold_results_dir)
            logging.info(f"🗑️ Deleted results for Fold {fold} (not Fold {SAVE_FOLD_NUMBER})")
    else:
        logging.info(f"✅ Keeping Fold {fold} as the fixed fold to save")

# Save the specified fold's results
target_fold_result_dir = f"{PREDICTION_DIR}/{FILE_NAME}/fold_{SAVE_FOLD_NUMBER}"
target_fold_weights_dir = f"{WEIGHTS_DIR}/{FILE_NAME}/fold_{SAVE_FOLD_NUMBER}"
os.makedirs(target_fold_result_dir, exist_ok=True)
os.makedirs(target_fold_weights_dir, exist_ok=True)

if os.path.exists(target_fold_weights_dir):
    # Save the target fold's predictions
    target_fold_data = torch.load(f'{data_config["cross_validation_data_dir"]}/cross_valid_fold_{SAVE_FOLD_NUMBER}.pt')
    target_fold_train_dataset = DrugResponseDataset(**target_fold_data['train'])
    target_fold_val_dataset = DrugResponseDataset(**target_fold_data['validation'])
    target_fold_train_loader = DataLoader(target_fold_train_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
    target_fold_val_loader = DataLoader(target_fold_val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
    
    # Load the best model
    target_fold_checkpoint = torch.load(f"{chpt_dir}/fold_{SAVE_FOLD_NUMBER}/best_model.pth")
    
    # Recreate model and load weights
    model = DrugResponseModel(
        pathway_gene_indices=pathway_gene_indices,
        gene_ffn_output_dim=GENE_FFN_OUTPUT_DIM,
        drug_ffn_output_dim=DRUG_FFN_OUTPUT_DIM,
        cross_attn_dim=CROSS_ATTN_EMBEDDING_DIM,
        final_dim=FINAL_EMBEDDING_DIM,
        max_gene_slots=MAX_GENE_SLOTS,
        drug_input_dim=DRUG_INPUT_DIM,
        isDiffer=IS_DIFFER,
        gene_ffn_hidden_dim=model_config['gene_ffn_hidden_dim'],
        drug_ffn_hidden_dim=model_config['drug_ffn_hidden_dim'],
        gene_ffn_dropout=model_config['gene_ffn_dropout'],
        drug_ffn_dropout=model_config['drug_ffn_dropout'],
        num_heads=model_config['num_heads'],
        depth=model_config['depth'],
        mlp_dropout=model_config['mlp_dropout'],
        final_dim_reduction_factor=model_config['final_dim_reduction_factor']
    ).to(DEVICE)
    model.load_state_dict(target_fold_checkpoint['model_state_dict'])
    
    model.eval()
    
    # Generate train predictions
    target_fold_train_actuals, target_fold_train_predictions, target_fold_train_samples = [], [], []
    with torch.no_grad():
        for batch in tqdm(target_fold_train_loader, desc=f"Generating Fold {SAVE_FOLD_NUMBER} train predictions"):
            outputs, loss, rmse, sample_indices, labels = process_batch(0, batch, 0, model, criterion, DEVICE, mode="val", is_best_epoch=False, save_weights=False)
            target_fold_train_actuals.extend(labels)
            target_fold_train_predictions.extend(outputs)
            target_fold_train_samples.extend(sample_indices)
    
    # Generate validation predictions
    target_fold_val_actuals, target_fold_val_predictions, target_fold_val_samples = [], [], []
    with torch.no_grad():
        for batch in tqdm(target_fold_val_loader, desc=f"Generating Fold {SAVE_FOLD_NUMBER} validation predictions"):
            outputs, loss, rmse, sample_indices, labels = process_batch(0, batch, 0, model, criterion, DEVICE, mode="val", is_best_epoch=False, save_weights=False)
            target_fold_val_actuals.extend(labels)
            target_fold_val_predictions.extend(outputs)
            target_fold_val_samples.extend(sample_indices)
    
    # Save the target fold train predictions
    torch.save({
        "actuals": target_fold_train_actuals,
        "predictions": target_fold_train_predictions,
        "drug_labels": target_fold_train_samples
    }, os.path.join(target_fold_result_dir, "train_results.pt"))
    
    # Save the target fold validation predictions
    torch.save({
        "actuals": target_fold_val_actuals,
        "predictions": target_fold_val_predictions,
        "drug_labels": target_fold_val_samples
    }, os.path.join(target_fold_result_dir, "val_results.pt"))
    
    logging.info(f"✅ Fold {SAVE_FOLD_NUMBER} train and validation results saved in {target_fold_result_dir}")
    
    # Move the target fold's weights → final location
    best_epoch_weights = f"{target_fold_weights_dir}/best_epoch"
    if os.path.exists(best_epoch_weights):
        for file in os.listdir(best_epoch_weights):
            src = os.path.join(best_epoch_weights, file)
            dst = os.path.join(target_fold_weights_dir, file)
            shutil.copy2(src, dst)
        logging.info(f"✅ Fold {SAVE_FOLD_NUMBER} weights saved in {target_fold_weights_dir}")
    
    # Clean up
    if os.path.exists(best_epoch_weights):
        shutil.rmtree(best_epoch_weights)
    current_epoch_dir = f"{target_fold_weights_dir}/current_epoch"
    if os.path.exists(current_epoch_dir):
        shutil.rmtree(current_epoch_dir)
else:
    logging.warning(f"⚠️ Fold {SAVE_FOLD_NUMBER} weights directory not found!")
