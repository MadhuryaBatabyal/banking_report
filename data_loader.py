import kagglehub
import os
import pandas as pd
import streamlit as st

def load_data():
    try:
        st.info("Downloading dataset from Kaggle (via kagglehub)...")
        path = kagglehub.dataset_download("ealaxi/paysim1")
        st.success("Dataset download complete.")

        file_path = os.path.join(path, "PS_20174392719_1491204439457_log.csv")
        st.info("Reading CSV file...")
        # Optional: read only a portion of the file for debugging
        df = pd.read_csv(file_path, nrows=10000)  # Use nrows if you want to limit memory/time
        st.success(f"CSV loaded successfully: {df.shape[0]} rows, {df.shape[1]} columns")
        return df

    except Exception as e:
        st.error(f"Failed to load data: {e}")
        raise
