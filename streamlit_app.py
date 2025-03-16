import streamlit as st
import pandas as pd
import torch
import os
import umap
import numpy as np
from collections import defaultdict
import plotly.express as px

def load_data_and_umap(
    directory, 
    embedding_type,         
    anlz_tissue_types, 
    anlz_drugs, 
    random_state=42
):
    # 1) Embedding 로드
    if embedding_type == "pathway":
        embedding = torch.load(os.path.join(directory, "epoch_merged_pathway_embedding.pt"))
    elif embedding_type == "drug":
        embedding = torch.load(os.path.join(directory, "epoch_merged_drug_embedding.pt"))
    else:
        raise ValueError("embedding_type must be either 'pathway' or 'drug'.")

    # samples (cell_line, drug) 정보
    samples = torch.load(os.path.join(directory, "epoch_merged_samples.pt"))

    selected_indices = []
    tissue_labels = []
    cell_line_list = []
    drug_list = []
    tissue_counts = defaultdict(int)
    drug_counts = defaultdict(int)

    # 2) 필터링
    for i, (cell_line, drug) in enumerate(samples):
        parts = cell_line.split("__")
        if len(parts) == 2:
            tissue = parts[1]
            # Tissue와 Drug가 관심 목록에 있는지 확인
            if tissue in anlz_tissue_types and drug in anlz_drugs:
                selected_indices.append(i)
                tissue_labels.append(tissue)
                cell_line_list.append(cell_line)
                drug_list.append(drug)
                tissue_counts[tissue] += 1
                drug_counts[drug] += 1

    # 3) UMAP
    selected_embeddings = embedding[selected_indices]
    reducer = umap.UMAP(n_components=2, random_state=random_state)
    emb_2d = reducer.fit_transform(selected_embeddings.numpy())

    # 4) DataFrame
    df = pd.DataFrame({
        "UMAP1": emb_2d[:, 0],
        "UMAP2": emb_2d[:, 1],
        "Tissue": tissue_labels,
        "Cell_Line": cell_line_list,
        "Drug": drug_list
    })

    return df, tissue_counts, drug_counts


# 메인 Streamlit 코드 시작
st.title("UMAP Visualization (2025.03.17)")

# 분석 대상 설정
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

# 디렉터리 설정
epoch1_dir = "regression/weights/20250314_13/epoch_1"
epoch50_dir = "regression/weights/20250314_13/epoch_50"

# Tissue별 고정 색상 (Plotly용)
tissue_color_map = {
    "Lung": "red",
    "Haematopoietic and Lymphoid": "blue",
    "Central Nervous System": "green",
    "Skin": "purple",
    "Breast": "orange"
}

# Tissue Legend 순서
tissue_order = [
    "Lung",
    "Haematopoietic and Lymphoid",
    "Central Nervous System",
    "Skin",
    "Breast"
]

# Drug Legend 순서
drug_order = anlz_drugs


# 1) Pathway Embedding
st.subheader("1. Pathway Embedding")

# (1-1) Pathway Epoch1 로드 & UMAP
df_p1, tissue_counts_p1, drug_counts_p1 = load_data_and_umap(
    directory=epoch1_dir,
    embedding_type="pathway",
    anlz_tissue_types=anlz_tissue_types,
    anlz_drugs=anlz_drugs,
    random_state=42
)

# (1-2) Pathway Epoch50 로드 & UMAP
df_p50, tissue_counts_p50, drug_counts_p50 = load_data_and_umap(
    directory=epoch50_dir,
    embedding_type="pathway",
    anlz_tissue_types=anlz_tissue_types,
    anlz_drugs=anlz_drugs,
    random_state=42
)

# ---- 첫 번째 줄: Tissue 색상 (좌=Epoch1, 우=Epoch50)
col1, col2 = st.columns(2)
with col1:
    fig_p1_tissue = px.scatter(
        df_p1,
        x="UMAP1",
        y="UMAP2",
        color="Tissue",
        hover_data=["Cell_Line", "Drug"],
        color_discrete_map=tissue_color_map,
        category_orders={"Tissue": tissue_order},
        title="Tissue Color (Epoch 1)"
    )
    st.plotly_chart(fig_p1_tissue)

with col2:
    fig_p50_tissue = px.scatter(
        df_p50,
        x="UMAP1",
        y="UMAP2",
        color="Tissue",
        hover_data=["Cell_Line", "Drug"],
        color_discrete_map=tissue_color_map,
        category_orders={"Tissue": tissue_order},
        title="Tissue Color (Epoch 50)"
    )
    st.plotly_chart(fig_p50_tissue)

# ---- 두 번째 줄: Drug 색상 (좌=Epoch1, 우=Epoch50)
col3, col4 = st.columns(2)

with col3:
    fig_p1_drug = px.scatter(
        df_p1,
        x="UMAP1",
        y="UMAP2",
        color="Drug",
        hover_data=["Cell_Line", "Tissue"],
        category_orders={"Drug": drug_order},
        title="Drug Color (Epoch 1)"
    )
    st.plotly_chart(fig_p1_drug)


with col4:
    fig_p50_drug = px.scatter(
        df_p50,
        x="UMAP1",
        y="UMAP2",
        color="Drug",
        hover_data=["Cell_Line", "Tissue"],
        category_orders={"Drug": drug_order},
        title="Drug Color (Epoch 50)"
    )
    st.plotly_chart(fig_p50_drug)


# ============================================
# 2) Drug Embedding
# ============================================
st.subheader("2. Drug Embedding")

# (2-1) Drug Epoch1 로드 & UMAP
df_d1, tissue_counts_d1, drug_counts_d1 = load_data_and_umap(
    directory=epoch1_dir,
    embedding_type="drug",
    anlz_tissue_types=anlz_tissue_types,
    anlz_drugs=anlz_drugs,
    random_state=42
)

# (2-2) Drug Epoch50 로드 & UMAP
df_d50, tissue_counts_d50, drug_counts_d50 = load_data_and_umap(
    directory=epoch50_dir,
    embedding_type="drug",
    anlz_tissue_types=anlz_tissue_types,
    anlz_drugs=anlz_drugs,
    random_state=42
)

# ---- 첫 번째 줄: Tissue 색상 (좌=Epoch1, 우=Epoch50)
col5, col6 = st.columns(2)

with col5:
    fig_d1_tissue = px.scatter(
        df_d1,
        x="UMAP1",
        y="UMAP2",
        color="Tissue",
        hover_data=["Cell_Line", "Drug"],
        color_discrete_map=tissue_color_map,
        category_orders={"Tissue": tissue_order},
        title="Tissue Color (Epoch 1)"
    )
    st.plotly_chart(fig_d1_tissue)

with col6:
    fig_d50_tissue = px.scatter(
        df_d50,
        x="UMAP1",
        y="UMAP2",
        color="Tissue",
        hover_data=["Cell_Line", "Drug"],
        color_discrete_map=tissue_color_map,
        category_orders={"Tissue": tissue_order},
        title="Tissue Color (Epoch 50)"
    )
    st.plotly_chart(fig_d50_tissue)

# ---- 두 번째 줄: Drug 색상 (좌=Epoch1, 우=Epoch50)
col7, col8 = st.columns(2)

with col7:
    fig_d1_drug = px.scatter(
        df_d1,
        x="UMAP1",
        y="UMAP2",
        color="Drug",
        hover_data=["Cell_Line", "Tissue"],
        category_orders={"Drug": drug_order},
        title="Drug Color (Epoch 1)"
    )
    st.plotly_chart(fig_d1_drug)

    st.write("**Tissue Counts**")
    for t, cnt in tissue_counts_d50.items():
        st.write(f"{t}: {cnt}")


with col8:
    fig_d50_drug = px.scatter(
        df_d50,
        x="UMAP1",
        y="UMAP2",
        color="Drug",
        hover_data=["Cell_Line", "Tissue"],
        category_orders={"Drug": drug_order},
        title="Drug Color (Epoch 1)"
    )
    st.plotly_chart(fig_d50_drug)
    
    st.write("**Drug Counts**")
    for d, cnt in drug_counts_d50.items():
        st.write(f"{d}: {cnt}")
