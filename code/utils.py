
import matplotlib.pyplot as plt
import os
import torch
import gc


def clear_cache():
    torch.cuda.empty_cache()
    gc.collect()  # Python의 Garbage Collector 호출


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

def plot_statics(file_name, train_losses, val_losses, train_accuracies, val_accuracies):
    plot_dir = './plots/' + file_name
    os.makedirs(plot_dir, exist_ok=True)

    # 데이터 길이 확인
    if len(train_losses) != len(val_losses) or len(train_accuracies) != len(val_accuracies):
        print("Warning: Mismatched lengths in training and validation data!")
    
    if len(train_losses) == 0 or len(val_losses) == 0:
        print("Error: No data to plot for losses!")
        return
    if len(train_accuracies) == 0 or len(val_accuracies) == 0:
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
        range(1, len(train_accuracies) + 1),  # x 값으로 epoch 범위 설정
        train_accuracies, val_accuracies,
        "Training and Validation Accuracy", "Epoch", "Accuracy",
        "Train Accuracy", "Validation Accuracy",
        os.path.join(plot_dir, "accuracy.png")
    )
    
