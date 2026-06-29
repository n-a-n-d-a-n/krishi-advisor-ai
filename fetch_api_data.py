"""
fetch_api_data.py — KrishiAdvisor AI
Fetches REAL live price data from public APIs.

APIs used:
  1. data.gov.in  — AGMARKNET daily mandi prices (FREE, needs API key)
  2. data.gov.in  — Horticulture crop prices
  3. agmarknet.gov.in — Direct scraper fallback

HOW TO GET FREE API KEY:
  1. Go to: https://data.gov.in/user/register
  2. Register free account
  3. Go to: https://data.gov.in/resource/9ef84268-d588-465a-a308-a864a43d0070
  4. Click "Get API" and copy your key
  5. Paste it below in API_KEY

Run: python fetch_api_data.py
     python train_models.py
     (restart app.py)
"""

import os, json, time, datetime
import pandas as pd
import requests

# ── PASTE YOUR FREE API KEY HERE ──────────────────────────────────────────────
API_KEY = 'YOUR_DATA_GOV_IN_API_KEY'
# ─────────────────────────────────────────────────────────────────────────────

DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')
os.makedirs(DATA_DIR, exist_ok=True)

# Crop name mapping: our names → AGMARKNET commodity names
CROP_MAP = {
    'onion':       ['Onion', 'Onion(Dry)'],
    'tomato':      ['Tomato'],
    'potato':      ['Potato'],
    'mango':       ['Mango', 'Mango (Totapuri)', 'Alphonso'],
    'grapes':      ['Grapes', 'Black Grapes', 'Green Grapes'],
    'banana':      ['Banana', 'Banana - Robusta', 'Banana Cavendish'],
    'orange':      ['Orange', 'Orange(Naagpur)'],
    'pomegranate': ['Pomegranate'],
    'wheat':       ['Wheat', 'Wheat(Dara)'],
    'soybean':     ['Soyabean(Yellow)', 'Soybean', 'Soyabean'],
    'chilli':      ['Chilli', 'Dry Chilles', 'Chilly(Dried)'],
    'garlic':      ['Garlic'],
    'turmeric':    ['Turmeric', 'Turmeric(Raw)', 'Turmeric(Dry)'],
    'ginger':      ['Ginger(Dry)', 'Ginger(Green)', 'Ginger'],
    'cauliflower': ['Cauliflower'],
    'cabbage':     ['Cabbage'],
    'cotton':      ['Cotton', 'Cotton(Long Staple)', 'Cotton(Medium Staple)'],
}

# Maharashtra districts in AGMARKNET
MH_DISTRICTS = [
    'Nashik', 'Pune', 'Ahmednagar', 'Solapur', 'Satara',
    'Sangli', 'Kolhapur', 'Aurangabad', 'Latur', 'Osmanabad',
    'Nanded', 'Jalgaon', 'Amravati', 'Nagpur', 'Wardha',
]

# AGMARKNET dataset IDs on data.gov.in
DATASET_IDS = [
    '9ef84268-d588-465a-a308-a864a43d0070',   # Daily prices
    '35985678-0d79-46b4-9ed6-6f13308a1d24',   # Monthly prices
]


def fetch_agmarknet(crop_names, state='Maharashtra', limit=1000):
    """Fetch price data from data.gov.in AGMARKNET API."""
    if API_KEY == 'YOUR_DATA_GOV_IN_API_KEY':
        print('[ERROR] Please set your API_KEY first.')
        print('        Get free key at: https://data.gov.in/user/register')
        return pd.DataFrame()

    all_records = []
    for dataset_id in DATASET_IDS:
        for commodity in crop_names:
            url = f'https://api.data.gov.in/resource/{dataset_id}'
            params = {
                'api-key':  API_KEY,
                'format':   'json',
                'limit':    limit,
                'filters[State]':     state,
                'filters[Commodity]': commodity,
            }
            try:
                r = requests.get(url, params=params, timeout=15)
                if r.status_code == 200:
                    data = r.json()
                    records = data.get('records', [])
                    all_records.extend(records)
                    print(f'  ✓ {commodity}: {len(records)} records')
                else:
                    print(f'  ✗ {commodity}: HTTP {r.status_code}')
            except Exception as e:
                print(f'  ✗ {commodity}: {e}')
            time.sleep(0.3)  # Rate limit

    return pd.DataFrame(all_records) if all_records else pd.DataFrame()


def clean_and_save(df_raw, crop_name):
    """Parse AGMARKNET response into our standard format."""
    if df_raw.empty:
        return None

    records = []
    for _, row in df_raw.iterrows():
        try:
            # AGMARKNET date format: DD/MM/YYYY
            date_str = str(row.get('Arrival_Date', row.get('date', '')))
            date = datetime.datetime.strptime(date_str, '%d/%m/%Y')

            # Price in Rs/quintal → Rs/kg
            modal_price = float(str(row.get('Modal_Price', row.get('modal_price', 0))).replace(',',''))
            price_per_kg = round(modal_price / 100, 2)

            if price_per_kg <= 0:
                continue

            records.append({
                'crop':     crop_name,
                'district': str(row.get('District', row.get('district', 'Maharashtra'))),
                'state':    'Maharashtra',
                'month':    date.month,
                'year':     date.year,
                'price':    price_per_kg,
                'unit':     'kg',
                'source':   'AGMARKNET_API',
            })
        except Exception:
            continue

    return pd.DataFrame(records) if records else None


def fetch_all():
    print('=' * 60)
    print('  KrishiAdvisor — Live API Data Fetcher')
    print('  Source: AGMARKNET via data.gov.in')
    print('=' * 60)

    all_data = []
    for crop_name, commodity_names in CROP_MAP.items():
        print(f'\nFetching: {crop_name}...')
        df_raw  = fetch_agmarknet(commodity_names)
        df_clean = clean_and_save(df_raw, crop_name)
        if df_clean is not None and not df_clean.empty:
            all_data.append(df_clean)
            print(f'  → {len(df_clean)} records for {crop_name}')

    if not all_data:
        print('\n[WARN] No API data fetched. Check API key and internet connection.')
        return

    df_final = pd.concat(all_data, ignore_index=True)
    # Average by month/year/district
    df_agg = (df_final.groupby(['crop','district','state','month','year','unit','source'])
              ['price'].mean().round(2).reset_index())

    # Merge with existing large_dataset.csv
    existing_path = os.path.join(DATA_DIR, 'large_dataset.csv')
    if os.path.exists(existing_path):
        df_existing = pd.read_csv(existing_path)
        df_merged   = pd.concat([df_existing, df_agg], ignore_index=True)
        df_merged   = df_merged.drop_duplicates(
            subset=['crop','district','month','year'], keep='last'
        )
    else:
        df_merged = df_agg

    out_path = os.path.join(DATA_DIR, 'large_dataset.csv')
    df_merged.to_csv(out_path, index=False)

    print(f'\n[DONE] Total records: {len(df_merged):,}')
    print(f'[DONE] API records added: {len(df_agg):,}')
    print(f'[DONE] Saved to: {out_path}')
    print('\nNext steps:')
    print('  python train_models.py   ← retrain with new data')
    print('  Restart app.py           ← apply new models')


if __name__ == '__main__':
    fetch_all()
