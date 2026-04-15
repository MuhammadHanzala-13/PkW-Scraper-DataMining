import pandas as pd
import requests
from bs4 import BeautifulSoup
import time
import random
import os

INPUT_FILE = "data/pakwheels_cars_raw.csv"
OUTPUT_FILE = "data/pakwheels_cars_raw.csv"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/124.0.0.0 Safari/537.36"
}

def enrich_data():
    if not os.path.exists(INPUT_FILE):
        print(f"File not found: {INPUT_FILE}")
        return

    df = pd.read_csv(INPUT_FILE)

    if 'enrich_status' not in df.columns: df['enrich_status'] = "Pending"
    if 'body_type' not in df.columns: df['body_type'] = "Unknown"
    if 'assembly' not in df.columns: df['assembly'] = "Unknown"
    if 'exterior_color' not in df.columns: df['exterior_color'] = "Unknown"
    if 'registered_city' not in df.columns: df['registered_city'] = "Unknown"
    if 'features' not in df.columns: df['features'] = ""

    # Respect previously enriched data before the new column existed
    already_done_mask = df['body_type'].notna() & (df['body_type'] != "Unknown")
    df.loc[already_done_mask, 'enrich_status'] = "Done"

    print(f"Loaded {len(df)} cars. Starting deep enrichment...")
    
    # We only process cars that are 'Pending'
    missing_mask = df['enrich_status'] != "Done"
    indices_to_enrich = df[missing_mask].index.tolist()

    print(f"Cars remaining to enrich: {len(indices_to_enrich)}")

    if len(indices_to_enrich) == 0:
        print("All cars are already enriched! You are good to go.")
        return

    save_counter = 0
    
    try:
        for idx in indices_to_enrich:
            url = df.loc[idx, 'url']
            print(f"  [{len(indices_to_enrich) - save_counter} remaining] Extracting: {df.loc[idx, 'title']}")
            
            try:
                # Polite delay to prevent IP Ban
                time.sleep(random.uniform(1.0, 2.5))
                
                response = requests.get(url, headers=HEADERS, timeout=15)
                if response.status_code == 200:
                    soup = BeautifulSoup(response.text, "lxml")
                    
                    # Parse the detail table
                    table = soup.find('table', class_='table table-bordered text-center table-engine-detail')
                    if table:
                        for td in table.find_all('td'):
                            parts = td.text.strip().split('\n')
                            if len(parts) >= 2:
                                val = parts[0].strip()
                                key = parts[-1].strip().replace(" ", "_").lower()
                                
                                # Update DataFrame where found (overriding 'Unknown')
                                if key == "body_type": df.at[idx, 'body_type'] = val
                                elif key == "assembly": df.at[idx, 'assembly'] = val
                                elif key == "exterior_color": df.at[idx, 'exterior_color'] = val
                                elif key == "registered_city": df.at[idx, 'registered_city'] = val
                                
                    # Extract the features list (Airbags, ABS, Sunroof etc.)
                    feature_ul = soup.find('ul', id='scroll_car_feature')
                    if feature_ul:
                        feature_items = [li.text.strip() for li in feature_ul.find_all('li')]
                        df.at[idx, 'features'] = ", ".join(feature_items)
                        
                    # Mark as successfully visited
                    df.at[idx, 'enrich_status'] = "Done"
                    
            except Exception as e:
                print(f"    [Error] Failed to fetch {url}: {e}")
                # We do NOT mark as "Done" so it can retry later

            save_counter += 1
            
            # Save every 20 cars so you don't lose data
            if save_counter % 20 == 0:
                df.to_csv(OUTPUT_FILE, index=False)
                print("  [CHECKPOINT] Progress safely saved to csv.")

    except KeyboardInterrupt:
        print("\n[STOPPED] You paused the script. Saving current progress...")
        
    finally:
        df.to_csv(OUTPUT_FILE, index=False)
        print("  [SAVED] Task complete or interrupted safely.")

if __name__ == "__main__":
    enrich_data()
