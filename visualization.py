import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from scipy.stats import pointbiserialr
import streamlit as st

def plot_transaction_type(df):
    try:
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.countplot(x="type", data=df, palette="viridis", ax=ax)
        ax.set_title("Distribution of Transaction Types")
        ax.set_xlabel("Transaction Type")
        ax.set_ylabel("Count")
        st.pyplot(fig)
    except Exception as e:
        st.error(f"Error in plot_transaction_type: {e}")

def plot_correlation_heatmap(df):
    try:
        corrmatrix = df.corr()
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(corrmatrix, annot=True, fmt=".2f", cmap="coolwarm", cbar=True, ax=ax)
        ax.set_title("Correlation Heatmap")
        st.pyplot(fig)
    except Exception as e:
        st.error(f"Error in plot_correlation_heatmap: {e}")

def plot_spearman_heatmap(df):
    try:
        paysimforspearman = pd.get_dummies(df.copy(), columns=["type"], drop_first=True)
        corrmatrix_spearman = paysimforspearman.corr(method="spearman")
        fig, ax = plt.subplots(figsize=(12, 10))
        sns.heatmap(corrmatrix_spearman, annot=True, fmt=".2f", cmap="coolwarm", cbar=True, ax=ax)
        ax.set_title("Spearman Correlation Heatmap for All Variables")
        st.pyplot(fig)
    except Exception as e:
        st.error(f"Error in plot_spearman_heatmap: {e}")

def plot_pointbiserial_heatmap(df):
    try:
        numericalcolsforpb = ["amount", "oldbalanceOrg", "newbalanceOrig", "oldbalanceDest", "newbalanceDest"]
        transactiontypes = df.type.unique()
        pbcorrelations = pd.DataFrame(index=transactiontypes, columns=numericalcolsforpb)
        for numcol in numericalcolsforpb:
            for transtype in transactiontypes:
                binarytype = (df.type == transtype).astype(int)
                corr, _ = pointbiserialr(binarytype, df[numcol])
                pbcorrelations.loc[transtype, numcol] = corr
        fig, ax = plt.subplots(figsize=(10,6))
        sns.heatmap(pbcorrelations.astype(float), annot=True, fmt=".2f", cmap="coolwarm", cbar=True, ax=ax)
        ax.set_title("Point-Biserial Correlation Heatmap")
        ax.set_xlabel("Numerical Features")
        ax.set_ylabel("Transaction Type")
        st.pyplot(fig)
    except Exception as e:
        st.error(f"Error in plot_pointbiserial_heatmap: {e}")

def plot_pointbiserial_fraud_bar(df):
    try:
        transactiontypes = df.type.unique()
        correlationresults = {}
        for transtype in transactiontypes:
            binarytype = (df.type == transtype).astype(int)
            corr, _ = pointbiserialr(binarytype, df.isFraud)
            correlationresults[transtype] = corr
        pbfraudcorrelations = pd.Series(correlationresults)
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(x=pbfraudcorrelations.index, y=pbfraudcorrelations.values, palette="viridis", ax=ax)
        ax.set_title("Point-Biserial Correlation Transaction Type vs isFraud")
        ax.set_xlabel("Transaction Type")
        ax.set_ylabel("Point-Biserial Correlation")
        ax.set_xticklabels(pbfraudcorrelations.index, rotation=45, ha="right")
        fig.tight_layout()
        st.pyplot(fig)
    except Exception as e:
        st.error(f"Error in plot_pointbiserial_fraud_bar: {e}")

def plot_kmeans_cluster(df):
    try:
        samplesize = 10000
        paysimsample = df.sample(n=min(samplesize, len(df)), random_state=42)
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.scatterplot(
            x="amount",
            y="oldbalanceOrg",
            hue="cluster",
            data=paysimsample,
            alpha=0.6,
            palette="viridis",
            ax=ax
        )
        ax.set_title(f"K-means Clustering Results: Amount vs Old Balance (Sampled Data n={samplesize})")
        ax.set_xlabel("Transaction Amount")
        ax.set_ylabel("Old Balance Origin")
        st.pyplot(fig)
    except Exception as e:
        st.error(f"Error in plot_kmeans_cluster: {e}")

def plot_pca_cluster(pca_df):
    try:
        samplesize = 10000
        pcasample = pca_df.sample(n=min(samplesize, len(pca_df)), random_state=42)
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.scatterplot(
            x="PC1",
            y="PC2",
            hue="cluster",
            data=pcasample,
            alpha=0.6,
            palette="viridis",
            ax=ax
        )
        ax.set_title(f"PCA Visualization of Clusters (Sampled Data n={samplesize})")
        ax.set_xlabel("Principal Component 1")
        ax.set_ylabel("Principal Component 2")
        st.pyplot(fig)
    except Exception as e:
        st.error(f"Error in plot_pca_cluster: {e}")

def plot_kprototypes_heatmap():
    st.info("plot_kprototypes_heatmap() not implemented yet")

def plot_kprototypes_bar():
    st.info("plot_kprototypes_bar() not implemented yet")

def plot_autoencoder_loss():
    st.info("plot_autoencoder_loss() not implemented yet")

def plot_reconstruction_error():
    st.info("plot_reconstruction_error() not implemented yet")

def plot_rbm_anomaly_distribution():
    st.info("plot_rbm_anomaly_distribution() not implemented yet")

def plot_famd_components():
    st.info("plot_famd_components() not implemented yet")
