import os
import streamlit as st
import kagglehub

def load_data():
    # Create .kaggle folder if it does not exist
    os.makedirs(os.path.expanduser("~/.kaggle"), exist_ok=True)
    # Write kaggle.json from secret
    kaggle_json_content = st.secrets["kaggle"]
    with open(os.path.expanduser("~/.kaggle/kaggle.json"), "w") as f:
        f.write(f'{{"username":"{kaggle_json_content["username"]}","key":"{kaggle_json_content["key"]}"}}')
    os.chmod(os.path.expanduser("~/.kaggle/kaggle.json"), 0o600)

    st.info("Downloading dataset from Kaggle (via kagglehub)...")
    path = kagglehub.dataset_download("ealaxi/paysim1")
    st.success("Dataset download complete.")

    file_path = os.path.join(path, "PS_20174392719_1491204439457_log.csv")
    st.info("Reading CSV file...")
    df = pd.read_csv(file_path, nrows=10000)
    st.success(f"CSV loaded successfully: {df.shape[0]} rows, {df.shape[1]} columns")
    return df
