import streamlit as st
import pandas as pd

DATA_URL = "https://raw.githubusercontent.com/MadhuryaBatabyal/banking_report/main/paysim.csv"

@st.cache_data
def load_data():
    df = pd.read_csv(DATA_URL)
    return df

st.title("PaySim Fraud Detection Visualization")
df = load_data()
st.success("Data loaded from PaySim GitHub/Kaggle link!")
st.dataframe(df.head(100))
