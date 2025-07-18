import matplotlib.pyplot as plt
import os
import torch
import gc
import numpy as np
import random

def clear_cache():
    torch.cuda.empty_cache()
    gc.collect()  # Python의 Garbage Collector 호출

def log_memory_summary():
    print(torch.cuda.memory_summary(device=torch.device("cuda:3")))

def save_plot(x, y1, y2, title, xlabel, ylabel, legend1, legend2, filepath):
    plt.figure()
    plt.plot(x, y1, label=legend1)
    plt.plot(x, y2, label=legend2)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.title(title)
    plt.savefig(filepath)
    plt.close()


def plot_statics(file_name, label, train_rmses, val_rmses):
    plot_dir = './plots/' + file_name
    os.makedirs(plot_dir, exist_ok=True)

    # 데이터 길이 확인
    if len(train_rmses) != len(val_rmses):
        print("Warning: Mismatched lengths in training and validation data!")
    
    if len(train_rmses) == 0 or len(val_rmses) == 0:
        print("Error: No data to plot for RMSEs!")
        return
    if len(train_rmses) == 0 or len(val_rmses) == 0:
        print("Error: No data to plot for RMSEs!")
        return
    
    # RMLSE Plot
    save_plot(
        range(1, len(train_rmses) + 1),  # x 값으로 epoch 범위 설정
        train_rmses, val_rmses,
        "Training and Validation RMSE", "Epoch", "RMSE",
        "Train RMSE", "Validation RMSE",
        os.path.join(plot_dir, f"{label}_RMSE.png")
    )

def set_seed(val):
    torch.manual_seed(val)
    torch.cuda.manual_seed(val)
    torch.cuda.manual_seed_all(val)
    np.random.seed(val)
    random.seed(val)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def plot_predictions(y_true, y_pred, save_path):
    plt.figure(figsize=(6,6))
    plt.scatter(y_true, y_pred, alpha=0.6)
    plt.plot([min(y_true), max(y_true)], [min(y_true), max(y_true)], 'r--')
    plt.xlabel("Actual Values")
    plt.ylabel("Predicted Values")
    plt.title("Actual vs Predicted")
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()