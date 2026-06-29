# database.py — KrishiAdvisor V2
# SQLite schema: users, crops, sales, stock, predictions

import sqlite3
import os
from datetime import datetime

DB_PATH = os.path.join(os.path.dirname(__file__), 'data', 'krishi.db')
os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)


def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db():
    conn = get_db()
    c = conn.cursor()

    # ── Users ──────────────────────────────────────────────────────────────────
    # Add lang column if it doesn't exist (handles old databases)
    try:
        c.execute("ALTER TABLE users ADD COLUMN lang TEXT DEFAULT 'mr'")
        conn.commit()
    except Exception:
        pass  # Column already exists

    c.execute('''CREATE TABLE IF NOT EXISTS users (
        id            INTEGER PRIMARY KEY AUTOINCREMENT,
        name          TEXT NOT NULL,
        phone         TEXT UNIQUE NOT NULL,
        village       TEXT,
        district      TEXT,
        password_hash TEXT NOT NULL,
        lang          TEXT DEFAULT 'mr',
        created_at    TEXT DEFAULT CURRENT_TIMESTAMP
    )''')

    # ── Crops master (open list) ───────────────────────────────────────────────
    c.execute('''CREATE TABLE IF NOT EXISTS crops (
        id         INTEGER PRIMARY KEY AUTOINCREMENT,
        name_en    TEXT UNIQUE NOT NULL,
        name_mr    TEXT,
        category   TEXT,
        unit       TEXT DEFAULT 'kg'
    )''')

    # ── Price history (for AI model training) ─────────────────────────────────
    c.execute('''CREATE TABLE IF NOT EXISTS price_history (
        id         INTEGER PRIMARY KEY AUTOINCREMENT,
        crop_id    INTEGER REFERENCES crops(id),
        month      INTEGER,
        year       INTEGER,
        price      REAL,
        district   TEXT,
        source     TEXT DEFAULT 'system'
    )''')

    # ── User stock (inventory) ─────────────────────────────────────────────────
    c.execute('''CREATE TABLE IF NOT EXISTS stock (
        id            INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id       INTEGER REFERENCES users(id),
        crop_id       INTEGER REFERENCES crops(id),
        quantity_kg   REAL NOT NULL,
        purchase_price REAL,
        production_cost REAL,
        harvest_date  TEXT,
        storage_type  TEXT DEFAULT 'home',
        notes         TEXT,
        status        TEXT DEFAULT 'stored',
        created_at    TEXT DEFAULT CURRENT_TIMESTAMP
    )''')

    # ── Sales history ──────────────────────────────────────────────────────────
    c.execute('''CREATE TABLE IF NOT EXISTS sales (
        id           INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id      INTEGER REFERENCES users(id),
        stock_id     INTEGER REFERENCES stock(id),
        crop_id      INTEGER REFERENCES crops(id),
        quantity_kg  REAL,
        sale_price   REAL,
        total_amount REAL,
        buyer        TEXT,
        sale_date    TEXT,
        profit       REAL,
        notes        TEXT,
        created_at   TEXT DEFAULT CURRENT_TIMESTAMP
    )''')

    # ── AI Predictions log ─────────────────────────────────────────────────────
    c.execute('''CREATE TABLE IF NOT EXISTS predictions (
        id              INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id         INTEGER REFERENCES users(id),
        crop_id         INTEGER REFERENCES crops(id),
        quantity        REAL,
        current_price   REAL,
        best_strategy   TEXT,
        expected_profit REAL,
        risk_level      TEXT,
        result_json     TEXT,
        created_at      TEXT DEFAULT CURRENT_TIMESTAMP
    )''')

    # ── Seed crop master data ──────────────────────────────────────────────────
    crops_seed = [
        ('onion','कांदा','vegetable'),('mango','आंबा','fruit'),
        ('grapes','द्राक्षे','fruit'),('tomato','टोमॅटो','vegetable'),
        ('potato','बटाटा','vegetable'),('banana','केळी','fruit'),
        ('orange','संत्रा','fruit'),('pomegranate','डाळिंब','fruit'),
        ('wheat','गहू','grain'),('soybean','सोयाबीन','oilseed'),
        ('cotton','कापूस','cash crop'),('sugarcane','ऊस','cash crop'),
        ('cauliflower','फुलकोबी','vegetable'),('cabbage','कोबी','vegetable'),
        ('brinjal','वांगी','vegetable'),('okra','भेंडी','vegetable'),
        ('chilli','मिरची','spice'),('garlic','लसूण','spice'),
        ('ginger','आले','spice'),('turmeric','हळद','spice'),
        ('watermelon','टरबूज','fruit'),('papaya','पपई','fruit'),
        ('guava','पेरू','fruit'),('lemon','लिंबू','fruit'),
        ('coconut','नारळ','fruit'),('strawberry','स्ट्रॉबेरी','fruit'),
        ('peas','वाटाणा','vegetable'),('carrot','गाजर','vegetable'),
        ('radish','मुळा','vegetable'),('spinach','पालक','vegetable'),
    ]
    c.executemany(
        'INSERT OR IGNORE INTO crops (name_en, name_mr, category) VALUES (?,?,?)',
        crops_seed
    )

    conn.commit()
    conn.close()
    print("[DB] Database initialized.")


def seed_price_history():
    """Seed 3 years of synthetic price history for all crops."""
    import numpy as np
    np.random.seed(42)

    # Base prices per crop (₹/kg approximate)
    base_prices = {
        'onion':10,'mango':35,'grapes':50,'tomato':12,'potato':10,
        'banana':15,'orange':30,'pomegranate':60,'wheat':22,'soybean':45,
        'cotton':65,'sugarcane':3,'cauliflower':20,'cabbage':15,'brinjal':18,
        'okra':25,'chilli':80,'garlic':70,'ginger':55,'turmeric':90,
        'watermelon':12,'papaya':25,'guava':30,'lemon':35,'coconut':20,
        'strawberry':120,'peas':35,'carrot':22,'radish':15,'spinach':20,
    }

    conn = get_db()
    c = conn.cursor()
    crops = c.execute('SELECT id, name_en FROM crops').fetchall()

    records = []
    for crop in crops:
        base = base_prices.get(crop['name_en'], 20)
        for year in range(2021, 2025):
            for month in range(1, 13):
                seasonal = base * 0.25 * np.sin(2 * np.pi * (month - 4) / 12)
                trend    = base * 0.05 * (year - 2021)
                noise    = np.random.normal(0, base * 0.08)
                price    = max(round(base + seasonal + trend + noise, 2), 1.0)
                records.append((crop['id'], month, year, price, 'Maharashtra'))

    c.executemany(
        'INSERT OR IGNORE INTO price_history (crop_id, month, year, price, district) VALUES (?,?,?,?,?)',
        records
    )
    conn.commit()
    conn.close()
    print(f"[DB] Seeded {len(records)} price records.")


if __name__ == '__main__':
    init_db()
    seed_price_history()
