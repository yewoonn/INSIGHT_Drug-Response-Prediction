import streamlit as st
import pandas as pd
import torch
import os
import umap
import numpy as np
from collections import defaultdict
import plotly.express as px
import plotly.graph_objects as go
from scipy.stats import gaussian_kde

# === 전역 설정 ===
anlz_tissue_types = [
    "Lung",
    "Haematopoietic and Lymphoid",
    "Central Nervous System",
    "Skin",
    "Breast"
]
anlz_drugs = [
    "Apitolisib", "Nilotinib", "ZSTK474", "BIX02189", "AS605240",
    "Foretinib", "AT7867", "AZD1332", "Olaparib", "ETP-45835"
]

# Tissue별 고정 색상 및 순서 (Plotly용)
tissue_color_map = {
    "Lung": "red",
    "Haematopoietic and Lymphoid": "blue",
    "Central Nervous System": "green",
    "Skin": "purple",
    "Breast": "orange"
}

tissue_order = [
    "Lung",
    "Haematopoietic and Lymphoid",
    "Central Nervous System",
    "Skin",
    "Breast"
]

# Drug Legend 순서
drug_order = anlz_drugs

# ----------------------------------------------------------------------
# 1. UMAP 계산 및 데이터프레임 생성 함수
# ----------------------------------------------------------------------
def load_data_and_umap(directory, embedding_type, anlz_tissue_types, anlz_drugs, random_state=42):
    # 1) Embedding 로드
    if embedding_type == "pathway":
        embedding = torch.load(os.path.join(directory, "epoch_merged_pathway_embedding.pt"))
    elif embedding_type == "drug":
        embedding = torch.load(os.path.join(directory, "epoch_merged_drug_embedding.pt"))
    else:
        raise ValueError("embedding_type must be either 'pathway' or 'drug'.")

    # samples (cell_line, drug) 정보 로드
    samples = torch.load(os.path.join(directory, "epoch_merged_samples.pt"))

    selected_indices = []
    tissue_labels = []
    cell_line_list = []
    drug_list = []
    tissue_counts = defaultdict(int)
    drug_counts = defaultdict(int)

    # 2) 관심 tissue와 drug로 필터링
    for i, (cell_line, drug) in enumerate(samples):
        parts = cell_line.split("__")
        if len(parts) == 2:
            tissue = parts[1]
            if tissue in anlz_tissue_types and drug in anlz_drugs:
                selected_indices.append(i)
                tissue_labels.append(tissue)
                cell_line_list.append(cell_line)
                drug_list.append(drug)
                tissue_counts[tissue] += 1
                drug_counts[drug] += 1

    # 3) UMAP 계산
    selected_embeddings = embedding[selected_indices]
    reducer = umap.UMAP(n_components=2, random_state=random_state)
    emb_2d = reducer.fit_transform(selected_embeddings.numpy())

    # 4) DataFrame 생성
    df = pd.DataFrame({
        "UMAP1": emb_2d[:, 0],
        "UMAP2": emb_2d[:, 1],
        "Tissue": tissue_labels,
        "Cell_Line": cell_line_list,
        "Drug": drug_list
    })

    return df, tissue_counts, drug_counts


# ----------------------------------------------------------------------
# 2. UMAP 플롯 생성 함수 (tissue, drug 색상)
# ----------------------------------------------------------------------
def create_umap_figures(directory, embedding_type, epoch_label):
    df, tissue_counts, drug_counts = load_data_and_umap(
        directory, embedding_type, anlz_tissue_types, anlz_drugs, random_state=42
    )
    # Tissue 색상 플롯
    fig_tissue = px.scatter(
        df,
        x="UMAP1",
        y="UMAP2",
        color="Tissue",
        hover_data=["Cell_Line", "Drug"],
        color_discrete_map=tissue_color_map,
        category_orders={"Tissue": tissue_order},
        title=f"Tissue Color ({epoch_label})"
    )
    # Drug 색상 플롯
    fig_drug = px.scatter(
        df,
        x="UMAP1",
        y="UMAP2",
        color="Drug",
        hover_data=["Cell_Line", "Tissue"],
        category_orders={"Drug": drug_order},
        title=f"Drug Color ({epoch_label})"
    )
    return df, fig_tissue, fig_drug, tissue_counts, drug_counts


# ----------------------------------------------------------------------
# 3-a. Flattened 임베딩 분포 (개별) 함수 (gaussian_kde 이용)
# ----------------------------------------------------------------------
def create_flattened_distribution_fig(directory, embedding_type, epoch_label, bw_method=0.1):
    if embedding_type == "pathway":
        embedding = torch.load(os.path.join(directory, "epoch_merged_pathway_embedding.pt"))
    elif embedding_type == "drug":
        embedding = torch.load(os.path.join(directory, "epoch_merged_drug_embedding.pt"))
    else:
        raise ValueError("embedding_type must be either 'pathway' or 'drug'.")

    embedding_np = embedding.numpy()
    flat = embedding_np.flatten()
    min_val, max_val = flat.min(), flat.max()
    margin = 0.1 * (max_val - min_val)
    clip_min = min_val - margin
    clip_max = max_val + margin

    x_grid = np.linspace(clip_min, clip_max, 200)
    kde = gaussian_kde(flat, bw_method=bw_method)
    y_values = kde(x_grid)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=x_grid,
            y=y_values,
            mode="lines",
            name=f"{embedding_type.capitalize()} Embedding"
        )
    )
    fig.update_layout(
        title=f"{embedding_type.capitalize()} Embedding Distribution ({epoch_label})",
        xaxis_title="Values",
        yaxis_title="Density"
    )
    return fig


# ----------------------------------------------------------------------
# 3-b. Flattened 임베딩 분포 (Pathway & Drug 동시에)
# ----------------------------------------------------------------------
def create_combined_flattened_distribution_fig(directory, epoch_label, bw_method=0.1):
    """
    Pathway & Drug 임베딩을 하나의 플롯에 겹쳐서 보여주는 Flattened Embedding Distribution
    + 각 임베딩 값 범위를 st.write로 출력
    """
    # 1) 임베딩 로드
    pathway_embedding = torch.load(os.path.join(directory, "epoch_merged_pathway_embedding.pt"))
    drug_embedding = torch.load(os.path.join(directory, "epoch_merged_drug_embedding.pt"))

    # 2) Flatten
    p_flat = pathway_embedding.numpy().flatten()
    d_flat = drug_embedding.numpy().flatten()

    # 값 범위 확인 (print or st.write)
    p_min, p_max = p_flat.min(), p_flat.max()
    d_min, d_max = d_flat.min(), d_flat.max()

    # 여기서 실제로 뷰에 표시하려면, 나중에 함수를 호출한 뒤에 st.write(...)를 해야 함
    # 함수 내부에서 할 수도 있지만, 보통 display_all_plots_for_date에서 호출 후에 st.write() 수행 가능
    # → 여기서는 "return"으로 값 범위를 함께 반환하면 됨
    # 예시로 그대로 함수 내부에서 st.write() 하려면 아래처럼:
    # (주의: Streamlit에서 함수가 여러 번 호출될 때 UI가 중복될 수 있음)
    # st.write(f"[Combined] Pathway min: {p_min}, max: {p_max}")
    # st.write(f"[Combined] Drug    min: {d_min}, max: {d_max}")

    # 3) x축 범위 (Pathway & Drug 통합 범위)
    combined_min = min(p_min, d_min)
    combined_max = max(p_max, d_max)
    margin = 0.1 * (combined_max - combined_min)
    clip_min = combined_min - margin
    clip_max = combined_max + margin

    x_grid = np.linspace(clip_min, clip_max, 200)

    # 4) KDE 계산
    kde_pathway = gaussian_kde(p_flat, bw_method=bw_method)
    kde_drug = gaussian_kde(d_flat, bw_method=bw_method)

    y_pathway = kde_pathway(x_grid)
    y_drug = kde_drug(x_grid)

    # 5) Plotly Figure 생성
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=x_grid,
            y=y_pathway,
            mode="lines",
            name="Pathway Embedding"
        )
    )
    fig.add_trace(
        go.Scatter(
            x=x_grid,
            y=y_drug,
            mode="lines",
            name="Drug Embedding"
        )
    )

    fig.update_layout(
        title=f"Combined Pathway & Drug Distribution ({epoch_label})",
        xaxis_title="Values",
        yaxis_title="Density"
    )

    # 임베딩 값 범위도 함께 반환하면, 호출하는 곳에서 st.write(...) 가능
    return fig, (p_min, p_max), (d_min, d_max)


# ----------------------------------------------------------------------
# 4. 날짜(또는 에폭)별 전체 플롯을 디스플레이하는 함수
# ----------------------------------------------------------------------
def display_all_plots_for_date(directory, epoch_label):
    st.header(f"Embedding Analysis for {epoch_label}")

    # [1] Pathway Embedding (UMAP)
    st.subheader("1. Pathway Embedding")
    _, fig_pathway_tissue, fig_pathway_drug, tissue_counts, drug_counts = create_umap_figures(directory, "pathway", epoch_label)
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(fig_pathway_tissue)
    with col2:
        st.plotly_chart(fig_pathway_drug)

    # [2] Drug Embedding (UMAP)
    st.subheader("2. Drug Embedding")
    _, fig_drug_tissue, fig_drug_drug, tissue_counts_d, drug_counts_d = create_umap_figures(directory, "drug", epoch_label)
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(fig_drug_tissue)
    with col2:
        st.plotly_chart(fig_drug_drug)

    # [3-a] Flattened Embedding Distribution (개별 Pathway & Drug)
    st.subheader("3-a. Flattened Embedding Distribution (Separate)")
    fig_pathway_flat = create_flattened_distribution_fig(directory, "pathway", epoch_label)
    fig_drug_flat = create_flattened_distribution_fig(directory, "drug", epoch_label)
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(fig_pathway_flat)
    with col2:
        st.plotly_chart(fig_drug_flat)

    # [3-b] Flattened Embedding Distribution (Pathway & Drug 겹침)
    st.subheader("3-b. Combined Flattened Embedding Distribution")
    fig_combined, (p_min, p_max), (d_min, d_max) = create_combined_flattened_distribution_fig(directory, epoch_label, bw_method=0.1)
    st.plotly_chart(fig_combined)

    # 임베딩 값의 min/max 범위 출력
    st.write(f"Pathway Embedding range: ({p_min:.6g}, {p_max:.6g})")
    st.write(f"Drug Embedding range:    ({d_min:.6g}, {d_max:.6g})")


# === 메인 Streamlit 앱 ===
st.title("UMAP Visualization by Date")

# 날짜(또는 에폭)별 디렉터리 매핑
# 원하는 만큼 추가할 수 있습니다.
directories = {
    "# Bug Fixed": "regression/weights/20250317_14/epoch_1",
    "# Bug Fixed": "regression/weights/20250317_14/epoch_15",
    "# Layer Norm Before Concat": "regression/weights/20250318_18/epoch_1",
    "# Layer Norm Before Concat": "regression/weights/20250318_18/epoch_15",
    "# Layer Norm After Concat" : "regression/weights/20250320_16/epoch_1",
    "# Layer Norm After Concat" : "regression/weights/20250320_16/epoch_11"
}

# 각 날짜별로 플롯 출력
for label, dir_path in directories.items():
    display_all_plots_for_date(dir_path, label)
