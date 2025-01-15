
import matplotlib.pyplot as plt
import os
import torch
import gc
import json


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

import os

class AttentionLogger:
    def __init__(self):
        """
        gene_records / sub_records 에
        (cell_line, drug, pathway, gene, substructure, weight) 형태의 튜플을 계속 쌓는다.
        """
        self.gene_records = []
        self.sub_records = []

    def add_gene_attention(
        self, sample_indices, pathway_id, gene_attention_weights,
        valid_gene_indices, set_global_ids
    ):
        """
        gene_attention_weights: [Batch, Num_Valid_Genes, Num_Substructures]
        sample_indices:         [Batch, 2]  # (cell_line_id, drug_id)
        valid_gene_indices:     List[int]
        set_global_ids:         List[int]
        """
        batch_size = gene_attention_weights.size(0)

        for b in range(batch_size):
            cell_line_id, drug_id = sample_indices[b]
            weight = gene_attention_weights[b,:,:]
            record = (
                 cell_line_id,
                 drug_id,
                 pathway_id,
                 valid_gene_indices,
                 set_global_ids,
                 weight
            )
            self.gene_records.append(record)


    def add_sub_attention(
        self, sample_indices, pathway_id, sub_attention_weights,
        valid_gene_indices, set_global_ids
    ):
        """
        sub_attention_weights: [Batch, Num_Substructures, Num_Valid_Genes]
        sample_indices:        [Batch, 2]  # (cell_line_id, drug_id)
        valid_gene_indices:    List[int]
        set_global_ids:        List[int]
        """
        batch_size = sub_attention_weights.size(0)

        for b in range(batch_size):
            cell_line_id, drug_id = sample_indices[b]
            weight = sub_attention_weights[b,:,:]
            record = (
                 cell_line_id,
                 drug_id,
                 pathway_id,
                 valid_gene_indices, # Gene
                 set_global_ids, #  Subs
                 weight
            )

            self.sub_records.append(record)



    def flush_to_json(self, file_name, epoch):
        # 파일명 설정
        attn_dir = f"weights/{file_name}/epoch_{epoch}"
        os.makedirs(attn_dir, exist_ok=True)

        gene_file = f'{attn_dir}/gene2sub_attn.json'
        sub_file = f'{attn_dir}/sub2gene_attn.json'

        # (1) gene_records 저장
        # 기존 데이터 로드
        if os.path.exists(gene_file):
            with open(gene_file, "r") as f:
                existing_gene_data = json.load(f)["gene_records"]
        else:
            existing_gene_data = []

        # 현재 데이터 추가
        new_gene_data = [
            {
                "cell_line": rec[0],
                "drug": rec[1],
                "pathway": rec[2],
                "valid_gene_indices": rec[3],
                "set_global_ids": rec[4],
                "weight": rec[5].tolist() 
            }
            for rec in self.gene_records
        ]
        combined_gene_data = existing_gene_data + new_gene_data

        # JSON 저장
        with open(gene_file, "w") as f:
            json.dump({"gene_records": combined_gene_data}, f, indent=4)

        # (2) sub_records 저장
        # 기존 데이터 로드
        if os.path.exists(sub_file):
            with open(sub_file, "r") as f:
                existing_sub_data = json.load(f)["sub_records"]
        else:
            existing_sub_data = []

        # 현재 데이터 추가
        new_sub_data = [
            {
                "cell_line": rec[0],
                "drug": rec[1],
                "pathway": rec[2],
                "valid_gene_indices": rec[3],
                "set_global_ids": rec[4],
                "weight": rec[5].tolist()  # 텐서를 리스트로 변환
            }
            for rec in self.sub_records
        ]
        combined_sub_data = existing_sub_data + new_sub_data

        # JSON 저장
        with open(sub_file, "w") as f:
            json.dump({"sub_records": combined_sub_data}, f, indent=4)


        # 버퍼를 비워줌
        self.gene_records.clear()
        self.sub_records.clear()
