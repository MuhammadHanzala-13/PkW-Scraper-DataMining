import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import joblib
import os

INPUT_FILE = "data/pakwheels_cars_processed.csv"
MODEL_OUTPUT = "car_price_model.pkl"

def train():
    print("=" * 60)
    print("  PakWheels | Model Training (Random Forest)")
    print("=" * 60)
    
    if not os.path.exists(INPUT_FILE):
        print(f"Error: {INPUT_FILE} not found. Run data engineering first.")
        return
        
    df = pd.read_csv(INPUT_FILE)
    
    # 1. Select features dynamically based on what exists
    potential_features = [
        "year", "car_age", "mileage_km", "engine_cc", "feature_count",
        "fuel_type_encoded", "transmission_encoded", "body_type_encoded",
        "assembly_encoded", "registered_city_encoded", "exterior_color_encoded",
        "brand_encoded", "model_encoded"
    ]
    
    features = [f for f in potential_features if f in df.columns]
    target = "price"
    
    if target not in df.columns:
        print(f"Error: Target '{target}' not found in dataset.")
        return
        
    if not features:
        print("Error: No features found in dataset.")
        return
        
    # Drop rows where any of these critical columns might still be NA
    df = df.dropna(subset=features + [target])
    
    X = df[features]
    y = df[target]
    
    print(f"Training on {len(df)} records with {len(features)} features...")
    
    # 2. Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 3. Model Initialization and Training
    # Random Forest is highly robust against outliers and non-linear data
    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    
    print("Training model... (this may take a few seconds)")
    model.fit(X_train, y_train)
    
    # 4. Evaluation
    predictions = model.predict(X_test)
    
    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    
    print("\n[RESULTS]")
    print(f"R-squared (Accuracy): {r2 * 100:.2f}%")
    print(f"Mean Absolute Error:  PKR {mae:,.0f}")
    
    # 5. Save Model
    joblib.dump(model, MODEL_OUTPUT)
    print(f"\n[SUCCESS] Model saved to {MODEL_OUTPUT}")
    
    # Print Feature Importances
    importances = pd.Series(model.feature_importances_, index=features).sort_values(ascending=False)
    print("\nTop 5 Most Important Features for Pricing:")
    for feat, imp in importances.head(5).items():
        print(f" - {feat}: {imp*100:.1f}%")

if __name__ == "__main__":
    train()
