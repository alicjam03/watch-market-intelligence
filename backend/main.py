#libraries
from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Watch Market Intelligence API")
origins = [
    "http://127.0.0.1:5500",
    "http://localhost:5500",
    "https://alicjam03.github.io",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins = origins,
    allow_credentials = True,
    allow_methods = ["*"],
    allow_headers = ["*"]
)


#load data and models
BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = BASE_DIR / "data" / "processed" / "merged_watches_with_clusters.csv"
MODEL_SCALER_PATH = BASE_DIR / "models" / "price_scaler.pkl"
MODEL_KMEANS_PATH = BASE_DIR / "models" / "price_kmeans.pkl"

df = pd.read_csv(DATA_PATH, low_memory=False)
scaler = joblib.load(MODEL_SCALER_PATH)
kmeans = joblib.load(MODEL_KMEANS_PATH)

#convert price to numeric
df["price_usd"] = pd.to_numeric(df["price_usd"], errors="coerce")

#human readable cluster names
cluster_labels = {
    0:"Mid Luxury",
    1:"Entry Luxury",
    2:"High End Luxury",
    3:"Ultra High End Luxury"
}

#input model
class PriceInput(BaseModel):
    price: float

#check
@app.get("/")
def root():
    return{"status": "API is running"}

@app.post("/price-position")
def price_position(data: PriceInput):
    price = data.price

    #prep features
    X = np.array([[price, np.log10(price)]])
    X_scaled = scaler.transform(X)

    #predict cluster
    cluster_id = int(kmeans.predict(X_scaled)[0])
    tier_name = cluster_labels.get(cluster_id, "Unknown")

    #percentile in the market
    percentile = float((df["price_usd"] < price).mean()*100)

    # business recommendation
    if percentile > 90:
        recommendation = "This price is higher than most of the market. Position as premium."
    elif percentile < 20:
        recommendation = "This price is very competitive. Make sure you are not underpricing"
    else:
        recommendation = "This price is well alligned with the current market"

    return {
        "input_price": price,
        "market_percentile": round(percentile, 2),
        "price_tier": tier_name,
        "recommendation": recommendation
    }

#import libraries for design insights
from typing import Optional
from fastapi import Query

@app.get("/design-insights")
def design_insights(
        min_price: float = Query(..., description="Minimum price in USD"),
        max_price: float = Query(..., description="Maximum price in USD"),
):
    # Ensure price is numeric
    local_df = df.copy()
    local_df["price_usd"] = pd.to_numeric(local_df["price_usd"], errors="coerce")

    data = local_df[
        (local_df["price_usd"] >= min_price) &
        (local_df["price_usd"] <= max_price)
    ].copy()

    total = len(data)
    if total == 0:
        return{"message" : "No data for this filter."}

    case_mat = data["case_material"].value_counts().head(5).to_dict()
    strap_mat = data["strap_material"].value_counts().head(5).to_dict()
    dial_col = data["dial_color"].value_counts().head(5).to_dict()


    diameters = data["case_diameter_mm"].dropna()
    diam_summary = {
        "min": float(diameters.min()) if not diameters.empty else None,
        "max": float(diameters.max()) if not diameters.empty else None,
        "mean": float(diameters.mean()) if not diameters.empty else None,
    }

    return {
        "total_watches": int(total),
        "top_case_materials": case_mat,
        "top_strap_materials": strap_mat,
        "top_dial_colours": dial_col,
        "case_diameter_stats": diam_summary,
    }

@app.get("/competitor-snapshot")
def competitor_snapshot(
        price:float,
):
    X = np.array([[price, np.log10(price)]])
    X_scaled = scaler.transform(X)
    cluster_id = int(kmeans.predict(X_scaled)[0])

    data = df[df["price_cluster"] == cluster_id].copy()

    if data.empty:
        return {"message": "No competitors found for this price tier"}

    lower = price * 0.8
    upper = price * 1.2
    nearby = data[(data["price_usd"] >= lower) & (data["price_usd"] <= upper)]

    top_brands_tier = data["brand"].value_counts().head(5).to_dict()
    top_brands_nearby = nearby["brand"].value_counts().head(5).to_dict()

    return {
        "price": price,
        "cluster_id": cluster_id,
        "price_tier": cluster_labels.get(cluster_id, "Unknown"),
        "total_in_tier": int(len(data)),
        "total_nearby_price": int(len(nearby)),
        "top_brands_in_tier": top_brands_tier,
        "top_brands_near_price": top_brands_nearby,
    }






