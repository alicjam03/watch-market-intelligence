import pandas as pd

# -------------------------------
# LOAD DATA
# -------------------------------
listings = pd.read_csv("../data/raw/Luxury watch.csv")
market = pd.read_csv("../data/raw/Watches.csv", low_memory=False)

# -------------------------------
# CLEAN LUXURY WATCH (DESIGN DATA)
# -------------------------------
listings.columns = listings.columns.str.lower().str.replace(" ", "_")

listings = listings.rename(columns={
    "price_(usd)": "price_usd",
    "case_material": "case_material",
    "strap_material": "strap_material",
    "movement_type": "movement_type",
    "dial_color": "dial_color",
    "case_diameter_(mm)": "case_diameter_mm",
    "water_resistance": "water_resistance",
    "power_reserve": "power_reserve",
    "complications": "complications",
})

listings = listings[[
    "brand", "model", "price_usd", "movement_type", "case_material",
    "strap_material", "dial_color", "case_diameter_mm",
    "water_resistance", "power_reserve", "complications"
]]

listings["source"] = "specs"

# convert price like "9,500" → 9500.0
listings["price_usd"] = (
    listings["price_usd"]
    .astype(str)
    .str.replace(",", "")
    .astype(float)
)

# -------------------------------
# CLEAN WATCHES (MARKET DATA)
# -------------------------------
# drop index column and the old 'condition' column – we will use 'cond'
market = market.drop(columns=["Unnamed: 0", "condition"], errors="ignore")

market = market.rename(columns={
    "brand": "brand",
    "model": "model",
    "price": "price_usd",
    "mvmt": "movement_type",
    "casem": "case_material",
    "bracem": "strap_material",
    "sex": "gender",
    "yop": "year_of_production",
    "cond": "condition",
    "size": "size_mm",
})

market = market[[
    "brand", "model", "price_usd", "movement_type", "case_material",
    "strap_material", "gender", "year_of_production",
    "condition", "size_mm"
]]

market["source"] = "listings"

# clean price: "12,345" or "$12,345" → 12345.0
market["price_usd"] = (
    market["price_usd"]
    .astype(str)
    .str.replace(",", "")
    .str.replace("$", "")
)
market["price_usd"] = pd.to_numeric(market["price_usd"], errors="coerce")

# -------------------------------
# COMBINE DATASETS
# -------------------------------
merged = pd.concat([market, listings], ignore_index=True)

# drop rows with no usable price
merged = merged.dropna(subset=["price_usd"])

merged.to_csv("../data/processed/merged_watches.csv", index=False)

print("✅ Merged dataset created")
print("Rows:", len(merged))
print("Columns:", merged.columns.tolist())
print(merged.head())
