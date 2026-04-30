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
        
    # 5. Make/Brand and Model: Extract from title
    def extract_brand_model(title):
        title = str(title)
        if pd.isna(title) or not title.strip():
            return pd.Series(["Unknown", "Unknown"])
            
        words = title.split()
        if not words:
            return pd.Series(["Unknown", "Unknown"])
            
        brand = words[0]
        
        # Find the year to determine where the model name ends
        year_idx = -1
        for i, word in enumerate(words):
            # Look for a 4-digit number representing a year
            if i > 0 and re.match(r'^(19|20)\d{2}$', word):
                year_idx = i
                break
                
        if year_idx > 1:
            model = " ".join(words[1:year_idx])
        elif len(words) > 1:
            # If no year found, just take the second word as model as a fallback
            model = words[1]
        else:
            model = "Unknown"
            
        return pd.Series([brand, model])

    if "title" in df.columns:
        df[["brand", "model"]] = df["title"].apply(extract_brand_model)
        
    print("[CLEAN] Data types cleaned. Extracted Brand and Model from title.")
    return df

def handle_missing_values(df):
    before = len(df)
    
    # 1. Drop records where Target Variable (Price) or extreme critical feature (Year) is missing
    critical = [c for c in ["price", "year"] if c in df.columns]
    df.dropna(subset=critical, inplace=True)
    
    # 2. Impute Numeric Variables with MEDIAN (robust to outliers)
    if "mileage_km" in df.columns:
        median_mileage = df["mileage_km"].median()
        df["mileage_km"] = df["mileage_km"].fillna(median_mileage)
        
    if "engine_cc" in df.columns:
        median_engine = df["engine_cc"].median()
        df["engine_cc"] = df["engine_cc"].fillna(median_engine)

    # 3. Impute Categorical Variables with MODE
    categorical_cols = ["fuel_type", "transmission", "city", "body_type", "assembly", "exterior_color", "registered_city"]
    for col in categorical_cols:
        if col in df.columns:
            # fill empty strings or Unknown with None first
            df[col] = df[col].replace(["", "Unknown"], pd.NA)
            mode_val = df[col].mode()
            if not mode_val.empty:
                df[col] = df[col].fillna(mode_val[0])
            else:
                df[col] = df[col].fillna("Other")
                
    print(f"[MISSING] Handled NAs. Dropped {before - len(df)} rows with missing critical tags, imputed the rest.")
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
    
    # Feature engineering for newly added features list
    if "features" in df.columns:
        df["features"] = df["features"].fillna("")
        df["feature_count"] = df["features"].apply(lambda x: len(x.split(',')) if len(x.strip()) > 0 else 0)

    print("[ENGINEER] Calculated age, logarithmic price, mileage rates, and feature count.")
    return df

def encode_categoricals(df):
    for col in ["fuel_type", "transmission", "city", "body_type", "assembly", "exterior_color", "registered_city"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip().str.capitalize()
            df[f"{col}_encoded"] = pd.Categorical(df[col]).codes
    
    if "brand" in df.columns:
        top_brands = df["brand"].value_counts().nlargest(15).index
        df["brand_clean"] = df["brand"].where(df["brand"].isin(top_brands), other="Other")
        df["brand_encoded"] = pd.Categorical(df["brand_clean"]).codes
        
    if "model" in df.columns:
        top_models = df["model"].value_counts().nlargest(40).index
        df["model_clean"] = df["model"].where(df["model"].isin(top_models), other="Other")
        df["model_encoded"] = pd.Categorical(df["model_clean"]).codes
        
    print("[ENCODE] Encoded categorical features, brand, and model.")
    return df

def run_pipeline():
    print("=" * 60)
    print("  PakWheels | Data Engineering Pipeline")
    print("=" * 60)
    try:
        df = load_raw(INPUT_FILE)
        df = clean_types(df)
        df = handle_missing_values(df)
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
