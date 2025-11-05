import streamlit as st
from data_loader import load_data
from preprocessing import clean_and_select_features
from clustering import perform_kmeans, compute_pca
from visualization import plot_transaction_type, plot_correlation_heatmap, plot_kmeans_scatter, plot_pca_scatter

def main():
    st.title("Financial Transaction Analysis Dashboard")

    # Load and preprocess data
    df = load_data()
    paysim_clean, paysim_selected = clean_and_select_features(df)

    # Perform clustering
    cluster_labels = perform_kmeans(paysim_selected, n_clusters=3)
    paysim_clean['cluster'] = cluster_labels

    # PCA for visualization
    pca_df = compute_pca(paysim_selected, paysim_clean['cluster'])

    col1, col2 = st.columns(2)

    with col1:
        st.header("Transaction Type Distribution")
        fig1 = plot_transaction_type(paysim_clean)
        st.pyplot(fig1)

    with col2:
        st.header("Correlation Heatmap")
        fig2 = plot_correlation_heatmap(paysim_selected)
        st.pyplot(fig2)

    st.header("Clustering Results")

    sample_size = 10000
    sample_df = paysim_clean.sample(n=min(sample_size, len(paysim_clean)), random_state=42)
    pca_sample_df = pca_df.sample(n=min(sample_size, len(pca_df)), random_state=42)

    st.plotly_chart(plot_kmeans_scatter(sample_df), use_container_width=True)
    st.plotly_chart(plot_pca_scatter(pca_sample_df), use_container_width=True)

if __name__ == "__main__":
    main()
