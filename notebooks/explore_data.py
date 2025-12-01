import pandas as pd

# Load the two datasets
listings = pd.read_csv("../data/raw/Luxury watch.csv")
watches = pd.read_csv("../data/raw/Watches.csv")

print("\n=== LISTINGS (Luxury watch.csv) HEAD ===")
print(listings.head())

print("\n=== WATCHES (Watches.csv) HEAD ===")
print(watches.head())

print("\n=== LISTINGS COLUMNS ===")
print(listings.columns.tolist())

print("\n=== WATCHES COLUMNS ===")
print(watches.columns.tolist())

print("\n=== LISTINGS INFO ===")
print(listings.info())

print("\n=== WATCHES INFO ===")
print(watches.info())
