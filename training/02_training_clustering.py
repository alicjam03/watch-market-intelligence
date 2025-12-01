#import libraries
from pyexpat import features

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import joblib

#load merged data
df = pd.read_csv("../data/processed/merged_watches.csv", low_memory=False)
print("Loaded dataframe type:", type(df))
print("Shape:", df.shape)

#using price so make sure its numeric
df["price_usd"] = pd.to_numeric(df["price_usd"], errors="coerce")

#build price features
price_df = df[["price_usd"]].copy()
print("price data before cleaning:", type(price_df))

#drop nans
price_df = price_df.dropna()
print("shape after cleaning:", price_df.shape)

#log transform price for outliers
price_df["log_price"] = np.log10(price_df["price_usd"])

#scale data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(price_df[["price_usd", "log_price"]])

#k-means clustering
kmeans = KMeans(n_clusters=4, random_state=42, n_init="auto")
clusters = kmeans.fit_predict(X_scaled)

#attach clusters to main df
df.loc[price_df.index, "price_cluster"] = clusters

#save models and data
joblib.dump(scaler, "../models/price_scaler.pkl")
joblib.dump(kmeans, "../models/price_kmeans.pkl")

df.to_csv("../data/processed/merged_watches_with_clusters.csv", index=False)

print("clustering done")
print(df["price_cluster"].value_counts(dropna=False))
