import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

def plot_transaction_type(df):
    fig, ax = plt.subplots(figsize=(8,5))
    sns.countplot(x='type', data=df, palette='viridis', ax=ax)
    ax.set_title("Distribution of Transaction Types")
    ax.set_xlabel("Transaction Type")
    ax.set_ylabel("Count")
    return fig

def plot_correlation_heatmap(df):
    corr_matrix = df.corr()
    fig, ax = plt.subplots(figsize=(8,6))
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', cbar=True, ax=ax)
    ax.set_title("Correlation Heatmap")
    return fig

def plot_kmeans_scatter(df):
    fig = px.scatter(df, x='amount', y='oldbalanceOrg', color='cluster',
                    title=f"K-Means Clustering: Amount vs Old Balance (Sampled n={len(df)})")
    return fig

def plot_pca_scatter(df):
    fig = px.scatter(df, x='PC1', y='PC2', color='cluster',
                    title=f"PCA Visualization of Clusters (Sampled n={len(df)})")
    return fig
