import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.decomposition import PCA
import os
import streamlit as st

os.environ["KAGGLE_USERNAME"] = st.secrets["KAGGLE_USERNAME"]
os.environ["KAGGLE_KEY"] = st.secrets["KAGGLE_KEY"]

# Now you can use kagglehub or kaggle CLI commands as usual
# Example using kaggle CLI
os.system('kaggle datasets download -d ealaxi/paysim1 -p ./data --unzip')

import pandas as pd
df = pd.read_csv('./data/PS_20174392719_1491204439457_log.csv')

# Data cleaning
paysim_clean = df.copy()
paysim_clean = paysim_clean.drop(['nameOrig', 'nameDest'], axis=1)

if paysim_clean.isnull().any().any():
    paysim_clean.dropna(inplace=True)

paysim_clean['type'] = paysim_clean['type'].astype('category')

# Feature selection (non-zero variance features)
numeric_cols = paysim_clean.select_dtypes(include=np.number)
selector = VarianceThreshold(threshold=0)
selector.fit(numeric_cols)
selected_features_mask = selector.get_support()
selected_features = numeric_cols.columns[selected_features_mask]
paysim_selected = paysim_clean[selected_features]

# Scale features for clustering
scaler = StandardScaler()
scaled_features = scaler.fit_transform(paysim_selected)

# Perform KMeans clustering
np.random.seed(42)
kmeans_result = KMeans(n_clusters=3, random_state=42, n_init=10)
kmeans_result.fit(scaled_features)

# Add cluster labels to dataframe
paysim_clean['cluster'] = kmeans_result.labels_

# PCA for cluster visualization
scaler_pca = StandardScaler()
scaled_features_pca = scaler_pca.fit_transform(paysim_selected)
pca = PCA(n_components=2)
pca_features = pca.fit_transform(scaled_features_pca)
pca_df = pd.DataFrame(data=pca_features, columns=['PC1', 'PC2'])
pca_df['cluster'] = paysim_clean['cluster']

# Streamlit UI
st.title("Financial Transaction Analysis Dashboard")

col1, col2 = st.columns(2)

with col1:
    st.header("Transaction Type Distribution")
    fig1, ax1 = plt.subplots(figsize=(8, 5))
    sns.countplot(x='type', data=paysim_clean, palette='viridis', ax=ax1)
    ax1.set_title("Distribution of Transaction Types")
    ax1.set_xlabel("Transaction Type")
    ax1.set_ylabel("Count")
    st.pyplot(fig1)

with col2:
    st.header("Correlation Heatmap")
    corr_matrix = paysim_selected.corr()
    fig2, ax2 = plt.subplots(figsize=(8, 6))
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', cbar=True, ax=ax2)
    ax2.set_title("Correlation Heatmap")
    st.pyplot(fig2)

st.header("Clustering Results")

sample_size = 10000
if 'cluster' in paysim_clean.columns:
    # Sampled scatter plot Amount vs Old Balance
    sample_df = paysim_clean.sample(n=min(sample_size, len(paysim_clean)), random_state=42)
    fig3 = px.scatter(sample_df, x='amount', y='oldbalanceOrg', color='cluster',
                      title=f"K-Means Clustering: Amount vs Old Balance (Sampled n={len(sample_df)})")
    st.plotly_chart(fig3, use_container_width=True)

    # PCA scatter plot
    pca_sample_df = pca_df.sample(n=min(sample_size, len(pca_df)), random_state=42)
    fig4 = px.scatter(pca_sample_df, x='PC1', y='PC2', color='cluster',
                      title=f"PCA Visualization of Clusters (Sampled n={len(pca_sample_df)})")
    st.plotly_chart(fig4, use_container_width=True)
else:
    st.warning("Clustering has not been performed yet. Cannot display cluster visualizations.")
