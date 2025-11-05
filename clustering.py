from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import numpy as np
import pandas as pd

def perform_kmeans(data, n_clusters=3):
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(data)

    np.random.seed(42)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    kmeans.fit(scaled_features)

    return kmeans.labels_

def compute_pca(data, cluster_labels):
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(data)

    pca = PCA(n_components=2)
    pca_features = pca.fit_transform(scaled_features)

    pca_df = pd.DataFrame(data=pca_features, columns=['PC1', 'PC2'])
    pca_df['cluster'] = cluster_labels.values if isinstance(cluster_labels, pd.Series) else cluster_labels

    return pca_df
