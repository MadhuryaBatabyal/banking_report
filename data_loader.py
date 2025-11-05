import kagglehub
import os
import pandas as pd

def load_data():
    path = kagglehub.dataset_download("ealaxi/paysim1")
    file_path = os.path.join(path, "PS_20174392719_1491204439457_log.csv")
    df = pd.read_csv(file_path)
    return df
