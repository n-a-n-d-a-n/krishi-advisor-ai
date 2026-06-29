#!/usr/bin/env python3
"""
data_fetcher.py — KrishiAdvisor AI
Fetches real Maharashtra crop prices from Government of India AGMARKNET API
Run: python data_fetcher.py
"""
import os, json, time, datetime, urllib.request, urllib.parse
import pandas as pd

OUT_FILE = os.path.join(os.path.dirname(__file__), 'data', 'maharashtra_crop_prices.csv')

# Free public API key from data.gov.in (works without registration for basic use)
# Get your own free key at: https://data.gov.in/user/register
API_KEY  = "579b464db66ec23bdd000001cdd3946e44ce4aab825208f2fb05d9e"
BASE_URL = "https://api.data.gov.in/resource/9ef84268-d588-465a-a308-a864a43d0070"

CROP_MAP = {
    'onion':'Onion','mango':'Mango','tomato':'Tomato','potato':'Potato',
    'grapes':'Grapes','banana':'Banana','orange':'Orange',
    'pomegranate':'Pomegranate','wheat':'Wheat','soybean':'Soyabean',
    'chilli':'Dry Chillies','garlic':'Garlic','turmeric':'Turmeric',
    'cauliflower':'Cauliflower','cabbage':'Cabbage','brinjal':'Brinjal',
    'okra':'Bhindi(Ladies Finger)','ginger':'Ginger(Dry)',
    'cotton':'Cotton','watermelon':'Water Melon','papaya':'Papaya',
}

def fetch_crop(our_name, api_name, state='Maharashtra', limit=500):
    params = urllib.parse.urlencode({
        'api-key': API_KEY, 'format': 'json', 'limit': limit,
        'filters[state.keyword]': state,
        'filters[commodity.keyword]': api_name,
    })
    url = BASE_URL + '?' + params
    records = []
    try:
        req  = urllib.request.Request(url, headers={'User-Agent': 'KrishiAdvisorAI/2.0'})
        resp = urllib.request.urlopen(req, timeout=15)
        data = json.loads(resp.read())
        for r in data.get('records', []):
            try:
                price = float(r.get('modal_price') or r.get('min_price') or 0)
                price_per_kg = round(price / 100, 2)
                ds = r.get('arrival_date','')
                if '/' in ds:
                    p = ds.split('/'); month,year = int(p[1]),int(p[2])
                else:
                    dt = datetime.datetime.strptime(ds[:10],'%Y-%m-%d'); month,year=dt.month,dt.year
                if price_per_kg > 0:
                    records.append({'crop':our_name,'month':month,'year':year,'price':price_per_kg})
            except: pass
        print(f"  OK {our_name:15s} {len(records)} records")
    except Exception as e:
        print(f"  -- {our_name:15s} Error: {e}")
    return records

def run():
    print("="*55)
    print("  KrishiAdvisor — AGMARKNET Data Fetcher (data.gov.in)")
    print("="*55)
    all_r = []
    for our,api in CROP_MAP.items():
        all_r.extend(fetch_crop(our, api))
        time.sleep(0.4)
    if not all_r:
        print("No data fetched. Check internet connection.")
        return
    df = pd.DataFrame(all_r)
    monthly = (df.groupby(['crop','year','month'])['price'].mean()
                 .reset_index().round({'price':2}))
    if os.path.exists(OUT_FILE):
        old = pd.read_csv(OUT_FILE)
        monthly = pd.concat([old, monthly]).drop_duplicates(
            subset=['crop','year','month'], keep='last')
    monthly.to_csv(OUT_FILE, index=False)
    print(f"\nSaved {len(monthly)} records to {OUT_FILE}")
    print("Now run: python train_models.py")

if __name__ == '__main__': run()
