"""
PakWheels Car Listings Fast Scraper
Collects thousands of records in minutes from search pages perfectly.
Includes all essential details: Title, Price, City, Year, Mileage, Fuel, CC, Transmission.
"""

import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import random
import os

BASE_URL = "https://www.pakwheels.com/used-cars/search/-/"
OUTPUT_FILE = "data/pakwheels_cars_raw.csv"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    )
}

def extract_listings_from_page(url):
    listings = []
    try:
        response = requests.get(url, headers=HEADERS, timeout=15)
        if response.status_code != 200:
            print(f"  [WARN] Failed to load {url} (Status {response.status_code})")
            return []
            
        soup = BeautifulSoup(response.text, "lxml")
        cards = soup.select('div.search-title')

        for card in cards:
            parent = card.parent.parent # div.well container
            if not parent: continue
            
            data = {}
            
            # Title & Link
            title_a = card.find('a')
            if title_a:
                data['title'] = title_a.text.strip()
                data['url'] = "https://www.pakwheels.com" + title_a['href']
                
            # Price
            price = parent.find('div', class_='price-details')
            if price:
                data['price_raw'] = price.text.strip()
                
            # Location / City
            city = parent.find('ul', class_='search-vehicle-info')
            if city and city.find('li'):
                 data['city'] = city.find('li').text.strip()
                 
            # Tech specs
            tech = parent.find('ul', class_='search-vehicle-info-2')
            if tech:
                 lis = tech.find_all('li')
                 if len(lis) > 0: data['year'] = lis[0].text.strip()
                 if len(lis) > 1: data['mileage'] = lis[1].text.strip()
                 if len(lis) > 2: data['fuel_type'] = lis[2].text.strip()
                 if len(lis) > 3: data['engine_cc'] = lis[3].text.strip()
                 if len(lis) > 4: data['transmission'] = lis[4].text.strip()
                 
            if 'title' in data and 'price_raw' in data:
                 listings.append(data)
                 
    except Exception as e:
        print(f"  [ERROR] on {url}: {e}")
        
    return listings

def scrape(total_pages=50):
    os.makedirs("data", exist_ok=True)
    all_records = []

    print("=" * 60)
    print("  PakWheels | Fast & Large Dataset Scraper")
    print(f"  Target: {total_pages} pages (~{total_pages * 30} full listings)")
    print("=" * 60)

    for page_num in range(1, total_pages + 1):
        url = f"{BASE_URL}?page={page_num}"
        print(f"\n[PAGE {page_num}/{total_pages}] Fetching: {url}")
        
        listings = extract_listings_from_page(url)
        if not listings:
            print("  [STOP] No listings found. Stopping.")
            break
            
        print(f"  Found {len(listings)} detailed listings.")
        all_records.extend(listings)
        
        # Save checkpoint
        pd.DataFrame(all_records).to_csv(OUTPUT_FILE, index=False)
        print(f"  [CHECKPOINT] {len(all_records)} total records saved.")
        
        # Polite delay
        time.sleep(random.uniform(1.5, 3.0))

    print("\n" + "=" * 60)
    print(f"  Scraping Complete. HUGE dataset of {len(all_records)} records saved to {OUTPUT_FILE}")
    print("=" * 60)

if __name__ == "__main__":
    scrape(total_pages=250) # Grabs approx 3000 cars!
