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
from feature_generation import (
    prepare_kprototypes_df,
    prepare_autoencoder_history,
    prepare_ae_eval_df,
    prepare_rbm_eval_df,
    prepare_famd_df
)

def main():
    st.title("Financial Transaction Analysis Dashboard")
    df = load_data()
    paysim_clean, paysim_selected = clean_and_select_features(df)
    cluster_labels = perform_kmeans(paysim_selected, n_clusters=3)
    paysim_clean['cluster'] = cluster_labels
    pca_df = compute_pca(paysim_selected, paysim_clean['cluster'])

    # Main analytics visualizations
    plot_transaction_type(paysim_clean)
    plot_correlation_heatmap(paysim_selected)
    plot_spearman_heatmap(paysim_clean)
    plot_pointbiserial_heatmap(paysim_clean)
    plot_pointbiserial_fraud_bar(paysim_clean)
    plot_kmeans_cluster(paysim_clean)
    plot_pca_cluster(pca_df)

    # Prepare advanced dataframes/variables before passing to plot functions
    clustercharacteristicskpsampled = prepare_kprototypes_df(paysim_clean)
    plot_kprototypes_heatmap(clustercharacteristicskpsampled)
    plot_kprototypes_bar(clustercharacteristicskpsampled)

    history = prepare_autoencoder_history(df)
    plot_autoencoder_loss(history)

    paysimsampledforaeeval = prepare_ae_eval_df(df)
    plot_reconstruction_error(paysimsampledforaeeval)

    paysimsampledforrbmeval = prepare_rbm_eval_df(df)
    plot_rbm_anomaly_distribution(paysimsampledforrbmeval)

    famdsample = prepare_famd_df(paysim_clean)
    plot_famd_components(famdsample)

if __name__ == "__main__":
    main()
