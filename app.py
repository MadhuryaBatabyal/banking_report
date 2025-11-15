import streamlit as st
from data_loader import load_data
from preprocessing import clean_and_select_features
from clustering import perform_kmeans, compute_pca
from visualization import (
    plot_transaction_type,
    plot_correlation_heatmap,
    plot_spearman_heatmap,
    plot_pointbiserial_heatmap,
    plot_pointbiserial_fraud_bar,
    plot_kmeans_cluster,
    plot_pca_cluster,
    plot_kprototypes_heatmap,
    plot_kprototypes_bar,
    plot_autoencoder_loss,
    plot_reconstruction_error,
    plot_rbm_anomaly_distribution,
    plot_famd_components
)

def main():
    st.title("Financial Transaction Analysis Dashboard")

    df = load_data()
    paysim_clean, paysim_selected = clean_and_select_features(df)
    cluster_labels = perform_kmeans(paysim_selected, n_clusters=3)
    paysim_clean['cluster'] = cluster_labels
    pca_df = compute_pca(paysim_selected, paysim_clean['cluster'])

    # All main visualizations can be called from here
    # (Assumes all plot_* functions in visualization.py)
    plot_transaction_type(paysim_clean)
    plot_correlation_heatmap(paysim_selected)
    plot_spearman_heatmap(paysim_clean)
    plot_pointbiserial_heatmap(paysim_clean)
    plot_pointbiserial_fraud_bar(paysim_clean)
    plot_kmeans_cluster(paysim_clean)
    plot_pca_cluster(pca_df)
    plot_kprototypes_heatmap()
    plot_kprototypes_bar()
    plot_autoencoder_loss()
    plot_reconstruction_error()
    plot_rbm_anomaly_distribution()
    plot_famd_components()

if __name__ == "__main__":
    main()
