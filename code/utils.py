
import matplotlib.pyplot as plt
import os
import torch
import gc


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

def plot_statics(file_name, epoch_nums, train_losses, val_losses, train_accuracies, val_accuracies):
    plot_dir = './plots/' + file_name
    os.makedirs(plot_dir, exist_ok=True)
    
    save_plot(
        range(1, epoch_nums),
        train_losses, val_losses,
        "Training and Validation Loss", "Epoch", "Loss",
        "Train Loss", "Validation Loss",
        os.path.join(plot_dir, "loss.png")
    )
    
    save_plot(
        range(1, epoch_nums),
        train_accuracies, val_accuracies,
        "Training and Validation Accuracy", "Epoch", "Accuracy",
        "Train Accuracy", "Validation Accuracy",
        os.path.join(plot_dir, "accuracy.png")
    )
    
