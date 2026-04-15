"""
PakWheels Cars - Data Engineering Pipeline
Input:  data/pakwheels_cars_raw.csv    (raw scraped data)
Output: data/pakwheels_cars_processed.csv (clean, feature-engineered data for modeling)

"""

import pandas as pd
import numpy as np
import os
import re

INPUT_FILE  = "data/pakwheels_cars_raw.csv"
OUTPUT_FILE = "data/pakwheels_cars_processed.csv"
CURRENT_YEAR = 2024

def load_raw(filepath):
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Raw data not found: {filepath}. Run pakwheels_scraper.py first.")
    df = pd.read_csv(filepath)
    print(f"[LOAD] {len(df)} records loaded from {filepath}.")
    return df

def clean_types(df):
    # 1. Price: handles "PKR 15.5 lacs", "PKR 2.3 crore", etc.
    def parse_price(val):
        val = str(val).lower()
        num = re.sub(r'[^\d.]', '', val)
        if not num: return np.nan
        num = float(num)
        if 'lacs' in val or 'lac' in val:
            return int(num * 100_000)
        elif 'crore' in val or 'crores' in val:
            return int(num * 10_000_000)
        return int(num)

    if "price_raw" in df.columns:
        df["price"] = df["price_raw"].apply(parse_price)

    # 2. Mileage: Removes 'km' and commas
    if "mileage" in df.columns:
        df["mileage_km"] = df["mileage"].astype(str).str.replace(r"[^\d]", "", regex=True)
        df["mileage_km"] = pd.to_numeric(df["mileage_km"], errors="coerce")
        df.drop(columns=["mileage"], inplace=True)

    # 3. Engine CC: Removes 'cc'
    if "engine_cc" in df.columns:
        df["engine_cc"] = pd.to_numeric(
            df["engine_cc"].astype(str).str.replace(r"[^\d]", "", regex=True),
            errors="coerce"
        )
        
    # 4. Year: Make numeric
    if "year" in df.columns:
        df["year"] = pd.to_numeric(df["year"], errors="coerce")
        
    # 5. Make/Brand: Extract from title
    if "title" in df.columns:
        df["brand"] = df["title"].apply(lambda x: str(x).split(' ')[0] if pd.notna(x) else "Unknown")
        
    print(f"[CLEAN] Data types cleaned and scaled.")
    return df

def drop_missing(df):
    before = len(df)
    critical = [c for c in ["price", "year"] if c in df.columns]
    df.dropna(subset=critical, inplace=True)
    print(f"[MISSING] Dropped {before - len(df)} rows missing critical logic fields.")
    return df

def prune_outliers(df):
    before = len(df)
    if "price" in df.columns:
        df = df[(df["price"] >= 100_000) & (df["price"] <= 100_000_000)]
    if "year" in df.columns:
        df = df[(df["year"] >= 1980) & (df["year"] <= CURRENT_YEAR)]
    if "mileage_km" in df.columns:
        df = df[(df["mileage_km"] > 0) & (df["mileage_km"] <= 1_000_000)]
    print(f"[OUTLIERS] Removed {before - len(df)} outliers.")
    return df

def engineer_features(df):
    if "year" in df.columns:
        df["car_age"] = CURRENT_YEAR - df["year"]
    if "price" in df.columns:
        df["price_log"] = np.log1p(df["price"])
    if "mileage_km" in df.columns and "car_age" in df.columns:
        df["mileage_per_year"] = df["mileage_km"] / (df["car_age"] + 1)
    if "mileage_km" in df.columns:
        df["is_high_mileage"] = (df["mileage_km"] > 150_000).astype(int)
    print("[ENGINEER] Calculated age, logarithmic price, and mileage rates.")
    return df

def encode_categoricals(df):
    for col in ["fuel_type", "transmission", "city", "body_type", "assembly"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip().str.capitalize()
            df[f"{col}_encoded"] = pd.Categorical(df[col]).codes
    
    if "brand" in df.columns:
        top = df["brand"].value_counts().nlargest(10).index
        df["brand_clean"] = df["brand"].where(df["brand"].isin(top), other="Other")
        
    print("[ENCODE] Encoded categorical features.")
    return df

def run_pipeline():
    print("=" * 60)
    print("  PakWheels | Data Engineering Pipeline")
    print("=" * 60)
    try:
        df = load_raw(INPUT_FILE)
        df = clean_types(df)
        df = drop_missing(df)
        df = prune_outliers(df)
        df = engineer_features(df)
        df = encode_categoricals(df)
        
        os.makedirs("data", exist_ok=True)
        df.to_csv(OUTPUT_FILE, index=False)
        print(f"\n[SUCCESS] {len(df)} processed records saved to {OUTPUT_FILE}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    run_pipeline()
