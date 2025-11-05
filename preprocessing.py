import pandas as pd
import numpy as np
from sklearn.feature_selection import VarianceThreshold

def clean_and_select_features(df):
    paysim_clean = df.copy()
    paysim_clean = paysim_clean.drop(['nameOrig', 'nameDest'], axis=1)

    if paysim_clean.isnull().any().any():
        paysim_clean.dropna(inplace=True)

    paysim_clean['type'] = paysim_clean['type'].astype('category')

    numeric_cols = paysim_clean.select_dtypes(include=np.number)
    selector = VarianceThreshold(threshold=0)
    selector.fit(numeric_cols)
    selected_features_mask = selector.get_support()
    selected_features = numeric_cols.columns[selected_features_mask]
    paysim_selected = paysim_clean[selected_features]

    return paysim_clean, paysim_selected
