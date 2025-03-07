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


def plot_statics(file_name, train_losses, val_losses, train_rmses, val_rmses):
    plot_dir = './plots/' + file_name
    os.makedirs(plot_dir, exist_ok=True)

    # 데이터 길이 확인
    if len(train_losses) != len(val_losses) or len(train_rmses) != len(val_rmses):
        print("Warning: Mismatched lengths in training and validation data!")
    
    if len(train_losses) == 0 or len(val_losses) == 0:
        print("Error: No data to plot for losses!")
        return
    if len(train_rmses) == 0 or len(val_rmses) == 0:
        print("Error: No data to plot for accuracies!")
        return

    # Loss Plot
    save_plot(
        range(1, len(train_losses) + 1),  # x 값으로 epoch 범위 설정
        train_losses, val_losses,
        "Training and Validation Loss", "Epoch", "Loss",
        "Train Loss", "Validation Loss",
        os.path.join(plot_dir, "loss.png")
    )
    
    # Accuracy Plot
    save_plot(
        range(1, len(train_rmses) + 1),  # x 값으로 epoch 범위 설정
        train_rmses, val_rmses,
        "Training and Validation Accuracy", "Epoch", "Accuracy",
        "Train Accuracy", "Validation Accuracy",
        os.path.join(plot_dir, "accuracy.png")
    )

def set_seed(val):
    torch.manual_seed(val)
    torch.cuda.manual_seed(val)
    torch.cuda.manual_seed_all(val)
    np.random.seed(val)
    random.seed(val)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False