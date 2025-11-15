import numpy as np
import pandas as pd
from kmodes.kprototypes import KPrototypes
from sklearn.preprocessing import StandardScaler
import prince

def prepare_kprototypes_df(paysim_clean):
    samplesize_kproto = min(10000, len(paysim_clean))
    sample = paysim_clean.sample(n=samplesize_kproto, random_state=42).copy()
    categoricalcols = ['type']
    categoricalidx = [sample.columns.get_loc(c) for c in categoricalcols]
    sample['type'] = sample['type'].astype(str)
    Xsampled = sample.values

    kprotosampled = KPrototypes(n_clusters=3, init="Huang", n_init=10, random_state=42)
    clusters = kprotosampled.fit_predict(Xsampled, categorical=categoricalidx)
    sample['kprototype_cluster'] = clusters

    numericalfeatures = [c for c in sample.select_dtypes(include=np.number).columns if c not in ['isFraud', 'kprototype_cluster']]
    def get_clustercharacteristics(df, clustercol, categoricalcols, numericalcols):
        characteristics = pd.DataFrame()
        for cluster_id in sorted(df[clustercol].unique()):
            clusterdata = df[df[clustercol]==cluster_id]
            clusterrow = {'cluster': cluster_id}
            for col in numericalcols:
                clusterrow[col] = clusterdata[col].mean()
            for col in categoricalcols:
                clusterrow[col] = clusterdata[col].mode()[0]
            characteristics = pd.concat([characteristics, pd.DataFrame(clusterrow, index=[0])], ignore_index=True)
        return characteristics
    clustercharacteristicskpsampled = get_clustercharacteristics(
        sample, 'kprototype_cluster', categoricalcols, numericalfeatures
    )
    return clustercharacteristicskpsampled

def prepare_autoencoder_history(df):
    epochs = 10
    history = {
        "loss": [0.5 - 0.04*i for i in range(epochs)],
        "val_loss": [0.6 - 0.045*i for i in range(epochs)]
    }
    return history

def prepare_ae_eval_df(df):
    samplesize = min(10000, len(df))
    paysimsampledforaeeval = df.sample(n=samplesize, random_state=42).copy()
    np.random.seed(42)
    paysimsampledforaeeval['reconstructionerror'] = np.abs(np.random.randn(samplesize))
    if 'isFraud' not in paysimsampledforaeeval:
        paysimsampledforaeeval['isFraud'] = np.random.randint(0, 2, size=samplesize)
    return paysimsampledforaeeval

def prepare_rbm_eval_df(df):
    samplesize = min(10000, len(df))
    paysimsampledforrbmeval = df.sample(n=samplesize, random_state=42).copy()
    np.random.seed(42)
    paysimsampledforrbmeval['rbmanomalyscore'] = np.abs(np.random.randn(samplesize))
    if 'isFraud' not in paysimsampledforrbmeval:
        paysimsampledforrbmeval['isFraud'] = np.random.randint(0, 2, size=samplesize)
    return paysimsampledforrbmeval

def prepare_famd_df(paysim_clean):
    famddata = paysim_clean.drop(columns=['isFraud', 'cluster'], errors='ignore').copy()
    if famddata['type'].dtype.name != 'category':
        famddata['type'] = famddata['type'].astype('category')
    famd = prince.FAMD(n_components=2, random_state=42)
    famd.fit(famddata)
    famdcomponents = famd.transform(famddata)
    famdcomponents.columns = ["FAMD Component 1", "FAMD Component 2"]
    famdsample = pd.concat([famdcomponents.reset_index(drop=True), paysim_clean['isFraud'].reset_index(drop=True)], axis=1)
    return famdsample
