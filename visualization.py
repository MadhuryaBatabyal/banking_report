import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from scipy.stats import pointbiserialr

def plot_transaction_type(df):
    plt.figure(figsize=(10, 6))
    sns.countplot(x="type", data=df, palette="viridis")
    plt.title("Distribution of Transaction Types")
    plt.xlabel("Transaction Type")
    plt.ylabel("Count")
    plt.show()

def plot_correlation_heatmap(df):
    corrmatrix = df.corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(corrmatrix, annot=True, fmt=".2f", cmap="coolwarm", cbar=True)
    plt.title("Correlation Heatmap")
    plt.show()

def plot_spearman_heatmap(df):
    paysimforspearman = pd.get_dummies(df.copy(), columns=["type"], drop_first=True)
    corrmatrix_spearman = paysimforspearman.corr(method="spearman")
    plt.figure(figsize=(12, 10))
    sns.heatmap(corrmatrix_spearman, annot=True, fmt=".2f", cmap="coolwarm", cbar=True)
    plt.title("Spearman Correlation Heatmap for All Variables")
    plt.show()

def plot_pointbiserial_heatmap(df):
    numericalcolsforpb = ["amount", "oldbalanceOrg", "newbalanceOrig", "oldbalanceDest", "newbalanceDest"]
    transactiontypes = df.type.unique()
    pbcorrelations = pd.DataFrame(index=transactiontypes, columns=numericalcolsforpb)
    for numcol in numericalcolsforpb:
        for transtype in transactiontypes:
            binarytype = (df.type == transtype).astype(int)
            corr, _ = pointbiserialr(binarytype, df[numcol])
            pbcorrelations.loc[transtype, numcol] = corr
    plt.figure(figsize=(10,6))
    sns.heatmap(pbcorrelations.astype(float), annot=True, fmt=".2f", cmap="coolwarm", cbar=True)
    plt.title("Point-Biserial Correlation Heatmap")
    plt.xlabel("Numerical Features")
    plt.ylabel("Transaction Type")
    plt.show()

def plot_pointbiserial_fraud_bar(df):
    transactiontypes = df.type.unique()
    correlationresults = {}
    for transtype in transactiontypes:
        binarytype = (df.type == transtype).astype(int)
        corr, _ = pointbiserialr(binarytype, df.isFraud)
        correlationresults[transtype] = corr
    pbfraudcorrelations = pd.Series(correlationresults)
    plt.figure(figsize=(10, 6))
    sns.barplot(x=pbfraudcorrelations.index, y=pbfraudcorrelations.values, palette="viridis")
    plt.title("Point-Biserial Correlation Transaction Type vs isFraud")
    plt.xlabel("Transaction Type")
    plt.ylabel("Point-Biserial Correlation")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.show()

def plot_kmeans_cluster(df):
    samplesize = 10000
    paysimsample = df.sample(n=min(samplesize, len(df)), random_state=42)
    plt.figure(figsize=(10, 8))
    sns.scatterplot(
        x="amount",
        y="oldbalanceOrg",
        hue="cluster",
        data=paysimsample,
        alpha=0.6,
        palette="viridis"
    )
    plt.title(f"K-means Clustering Results: Amount vs Old Balance (Sampled Data n={samplesize})")
    plt.xlabel("Transaction Amount")
    plt.ylabel("Old Balance Origin")
    plt.show()

def plot_pca_cluster(pca_df):
    samplesize = 10000
    pcasample = pca_df.sample(n=min(samplesize, len(pca_df)), random_state=42)
    plt.figure(figsize=(10, 8))
    sns.scatterplot(
        x="PC1",
        y="PC2",
        hue="cluster",
        data=pcasample,
        alpha=0.6,
        palette="viridis"
    )
    plt.title(f"PCA Visualization of Clusters (Sampled Data n={samplesize})")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.show()

# Write similar functions for kprototypes, autoencoder, rbm, famd, etc.

