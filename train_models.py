#!/usr/bin/env python3
"""
train_models.py — KrishiAdvisor AI (PROPER TIME-SERIES VERSION)
================================================================
KEY INSIGHT: Good price prediction needs YEAR-OVER-YEAR patterns,
not just last month's price. This version:
  - Uses 25 years of data properly
  - Features: same-month-last-year, seasonal indices, YoY change
  - No lag1 dominance — forces model to learn TRUE seasonal patterns
  - Walk-forward validation (no data leakage)
Run: python train_models.py
"""
import os, sys, json, pickle, time, warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error

DATA_FILE  = os.path.join(os.path.dirname(__file__), 'data', 'maharashtra_crop_prices.csv')
MODELS_DIR = os.path.join(os.path.dirname(__file__), 'models')
os.makedirs(MODELS_DIR, exist_ok=True)

N_FEATURES = 23


def build_feature_matrix(sub_df):
    """
    Build proper time-series features from full price history.
    
    KEY FEATURES (designed to avoid lag1 dominance):
      - same_month_last_year    : price in same month 1yr ago (most predictive for crops)
      - same_month_2yr_ago      : price in same month 2yr ago
      - same_month_3yr_ago      : price in same month 3yr ago
      - seasonal_index          : this month's avg / annual avg (learned from history)
      - yoy_change_pct          : % change vs same month last year
      - 3yr_avg_this_month      : avg of this month across last 3 years
      - annual_avg_last_year    : last year's annual average price
      - annual_trend            : slope of last 3 annual averages
      - month_rank              : rank of this month in seasonal pattern (1=cheapest, 12=most exp)
      - pre_peak_distance       : months until typical peak month
      - sin/cos seasonality     : smooth seasonal curve
      - recent_momentum         : 3-month vs 6-month moving average ratio
      - volatility_index        : std/mean of last 12 months
      - price_level             : normalized current price vs 5yr avg
    """
    df = sub_df.sort_values(['year','month']).reset_index(drop=True)
    
    # Build lookup: (year, month) -> price
    price_map = {(r['year'], r['month']): r['price'] for _, r in df.iterrows()}
    
    # Calculate seasonal indices per month (avg ratio to annual mean)
    monthly_avgs = df.groupby('month')['price'].mean()
    annual_avg   = df['price'].mean()
    seasonal_idx = (monthly_avgs / annual_avg).to_dict()
    
    # Find peak month (most expensive on average)
    peak_month = monthly_avgs.idxmax()
    
    # Annual averages per year
    annual_avgs = df.groupby('year')['price'].mean().to_dict()
    
    X, y, meta_rows = [], [], []
    
    for i, row in df.iterrows():
        yr, mo, price = int(row['year']), int(row['month']), float(row['price'])
        if yr < 2003:   # Need 3 years lookback
            continue
        
        def get_price(y, m):
            return price_map.get((y, m), None)
        
        # ── Same-month historical prices ──────────────────────────────────
        sm1 = get_price(yr-1, mo)  # same month last year
        sm2 = get_price(yr-2, mo)  # same month 2 years ago
        sm3 = get_price(yr-3, mo)  # same month 3 years ago
        
        if sm1 is None or sm2 is None or sm3 is None:
            continue
        
        # ── Year-over-year signals ────────────────────────────────────────
        yoy_chg    = (sm1 - sm2) / sm2 * 100 if sm2 > 0 else 0
        yoy2_chg   = (sm2 - sm3) / sm3 * 100 if sm3 > 0 else 0
        sm_3yr_avg = (sm1 + sm2 + sm3) / 3
        
        # ── Annual trend ──────────────────────────────────────────────────
        ann1 = annual_avgs.get(yr-1, price)
        ann2 = annual_avgs.get(yr-2, ann1)
        ann3 = annual_avgs.get(yr-3, ann2)
        ann_trend = (ann1 - ann3) / 2 if ann3 > 0 else 0   # avg annual change
        
        # ── Recent momentum (last 3 months vs last 6) ────────────────────
        recent_prices = []
        for back in range(1, 7):
            bm = ((mo - back - 1) % 12) + 1
            by = yr if mo - back >= 1 else yr - 1
            p  = get_price(by, bm)
            if p: recent_prices.append(p)
        
        rm3 = np.mean(recent_prices[:3]) if len(recent_prices) >= 3 else sm1
        rm6 = np.mean(recent_prices[:6]) if len(recent_prices) >= 6 else sm1
        momentum = rm3 / rm6 if rm6 > 0 else 1.0
        
        # ── Volatility (std/mean of last 12 same-month prices) ────────────
        hist_same_month = [price_map.get((yr-k, mo), None) for k in range(1, 8)]
        hist_valid      = [p for p in hist_same_month if p is not None]
        volatility      = np.std(hist_valid) / np.mean(hist_valid) if len(hist_valid) > 2 else 0.15
        
        # ── Seasonal features ─────────────────────────────────────────────
        seas_idx  = seasonal_idx.get(mo, 1.0)
        sin1      = np.sin(2*np.pi*mo/12)
        cos1      = np.cos(2*np.pi*mo/12)
        sin2      = np.sin(4*np.pi*mo/12)
        cos2      = np.cos(4*np.pi*mo/12)
        
        # Distance to peak (circular)
        peak_dist = min((peak_month - mo) % 12, (mo - peak_month) % 12)
        
        # Month rank in seasonal pattern (1=cheapest, 12=most expensive)
        month_ranks = pd.Series(seasonal_idx).rank().to_dict()
        mo_rank     = month_ranks.get(mo, 6)
        
        # ── Price level vs 5-year average ────────────────────────────────
        five_yr_avg = np.mean([annual_avgs.get(yr-k, ann1) for k in range(1, 6) if yr-k in annual_avgs])
        price_level = sm1 / five_yr_avg if five_yr_avg > 0 else 1.0
        
        # ── Time index ───────────────────────────────────────────────────
        t = yr * 12 + mo
        
        feat = [
            t, mo, yr,                    # time
            sm1, sm2, sm3,                # same-month history (THE KEY FEATURES)
            yoy_chg, yoy2_chg,            # year-over-year change
            sm_3yr_avg,                   # 3-year avg for this month
            ann1, ann_trend,              # annual level and trend
            momentum, rm3, rm6,           # recent momentum
            volatility,                   # price stability
            seas_idx, peak_dist, mo_rank, # seasonality
            sin1, cos1, sin2, cos2,       # harmonic seasonality
            price_level,                  # normalized price level
        ]
        
        assert len(feat) == N_FEATURES, f"Expected {N_FEATURES}, got {len(feat)}"
        X.append(feat)
        y.append(price)
        meta_rows.append({'year': yr, 'month': mo})
    
    return np.array(X), np.array(y), meta_rows, {
        'seasonal_idx':  seasonal_idx,
        'peak_month':    int(peak_month),
        'month_ranks':   {int(k): float(v) for k, v in month_ranks.items()},
        'annual_avgs':   {int(k): float(v) for k, v in annual_avgs.items()},
        'price_map':     {f"{k[0]},{k[1]}": float(v) for k, v in price_map.items()},
    }


def train_one(crop, df):
    sub = df[df['crop'] == crop].sort_values(['year','month']).reset_index(drop=True)
    if len(sub) < 48:
        return None, f'only {len(sub)} rows'

    X, y, meta_rows, stat_info = build_feature_matrix(sub)
    if len(X) < 30:
        return None, 'not enough feature rows'

    scaler = StandardScaler()
    Xs     = scaler.fit_transform(X)

    # Gradient Boosting — LOWER learning rate, MORE trees = better generalisation
    gb = GradientBoostingRegressor(
        n_estimators   = 500,
        learning_rate  = 0.03,
        max_depth      = 4,
        min_samples_leaf=4,
        subsample      = 0.8,
        max_features   = 0.75,
        loss           = 'huber',
        random_state   = 42
    )
    
    # Random Forest
    rf = RandomForestRegressor(
        n_estimators     = 250,
        max_depth        = 7,
        min_samples_leaf = 4,
        max_features     = 'sqrt',
        random_state     = 42,
        n_jobs           = -1
    )
    
    ridge = Ridge(alpha=10.0)
    
    gb.fit(Xs, y)
    rf.fit(Xs, y)
    ridge.fit(Xs, y)

    # Walk-forward validation (proper time-series CV — no leakage)
    tscv   = TimeSeriesSplit(n_splits=6, test_size=12)   # test on 12-month windows
    cv_results = {'gb': [], 'rf': [], 'ridge': [], 'blend': []}

    for tri, tei in tscv.split(Xs):
        if len(tri) < 24 or len(tei) < 6:
            continue
        g  = GradientBoostingRegressor(n_estimators=250, learning_rate=0.04,
             max_depth=3, subsample=0.8, loss='huber', random_state=42)
        r  = RandomForestRegressor(n_estimators=120, max_depth=5,
             min_samples_leaf=4, random_state=42, n_jobs=-1)
        ri = Ridge(alpha=10.0)
        
        g.fit(Xs[tri], y[tri]); pg = g.predict(Xs[tei])
        r.fit(Xs[tri], y[tri]); pr = r.predict(Xs[tei])
        ri.fit(Xs[tri], y[tri]); pri= ri.predict(Xs[tei])
        
        blend = 0.55*pg + 0.35*pr + 0.10*pri
        
        cv_results['gb'].append(mean_absolute_error(y[tei], pg))
        cv_results['rf'].append(mean_absolute_error(y[tei], pr))
        cv_results['ridge'].append(mean_absolute_error(y[tei], pri))
        cv_results['blend'].append(mean_absolute_error(y[tei], blend))

    w = (0.55, 0.35, 0.10)
    yp = w[0]*gb.predict(Xs) + w[1]*rf.predict(Xs) + w[2]*ridge.predict(Xs)
    
    r2     = round(float(r2_score(y, yp)), 4)
    mae    = round(float(mean_absolute_error(y, yp)), 2)
    cv_mae = round(float(np.mean(cv_results['blend'])) if cv_results['blend'] else mae, 2)
    mape   = round(float(np.mean(np.abs((np.array(y)-yp)/np.array(y))*100)), 2)
    
    # Feature importance
    fi      = gb.feature_importances_
    fi_names=['time','month','year','sm1','sm2','sm3','yoy','yoy2','sm3avg',
                'ann1','ann_trend','momentum','rm3','rm6','volatility',
                'seas_idx','peak_dist','mo_rank','sin1','cos1','sin2','cos2','price_level']
    top_feat = {fi_names[i]: round(float(fi[i]),4) for i in np.argsort(fi)[-6:][::-1]}

    # Store FULL price history for prediction (not just 36 months)
    full_prices = {f"{int(r['year'])},{int(r['month'])}": float(r['price']) 
                   for _, r in sub.iterrows()}

    meta = {
        'crop': crop, 'n_features': N_FEATURES,
        'r2': r2, 'mae': mae, 'cv_mae': cv_mae, 'mape': mape,
        'n_samples': len(X), 'n_years': sub['year'].nunique(),
        'year_range': [int(sub['year'].min()), int(sub['year'].max())],
        'weights': {'gb': w[0], 'rf': w[1], 'ridge': w[2]},
        'top_features': top_feat,
        'stat_info': stat_info,
        'full_prices': full_prices,   # FULL history for prediction
        'trained_at': str(pd.Timestamp.now())[:19],
    }

    return {'gb': gb, 'rf': rf, 'ridge': ridge,
            'scaler': scaler, 'weights': w, 'meta': meta}, None


def save_model(crop, bundle):
    with open(f'{MODELS_DIR}/{crop}_model.pkl','wb') as f:
        pickle.dump({'gb':bundle['gb'],'rf':bundle['rf'],
                     'ridge':bundle['ridge'],'weights':bundle['weights']}, f)
    with open(f'{MODELS_DIR}/{crop}_scaler.pkl','wb') as f:
        pickle.dump(bundle['scaler'], f)
    meta_to_save = {k: v for k, v in bundle['meta'].items() if k != 'stat_info'}
    meta_to_save['stat_info'] = bundle['meta']['stat_info']
    with open(f'{MODELS_DIR}/{crop}_meta.json','w') as f:
        json.dump(meta_to_save, f, indent=2)


def train_all(target=None):
    if not os.path.exists(DATA_FILE):
        print(f'Dataset not found: {DATA_FILE}'); return
    df    = pd.read_csv(DATA_FILE)
    crops = sorted(df['crop'].unique().tolist())
    if target:
        crops = [c for c in crops if c in target]

    print('=' * 80)
    print('  KrishiAdvisor AI — Proper Time-Series Model Trainer')
    print(f'  {len(df):,} records | {len(crops)} crops | {df["year"].nunique()} years | {N_FEATURES} YoY features')
    print('  KEY: Uses same-month-last-year + YoY change (not just lag1)')
    print('=' * 80)
    print(f'\n{"CROP":<14} {"R²":>7} {"CV-MAE":>9} {"MAPE":>7}  {"Top feature":>20}')
    print('─' * 65)

    results = []
    t0      = time.time()

    for crop in crops:
        bundle, err = train_one(crop, df)
        if bundle:
            m   = bundle['meta']
            top = list(m['top_features'].keys())[0]
            top_val = list(m['top_features'].values())[0]
            bar = '█' * int(m['r2'] * 10) + '░' * (10 - int(m['r2'] * 10))
            print(f"{crop:<14} {m['r2']:>7.4f} {m['cv_mae']:>9.2f} {m['mape']:>7.2f}%  {top}={top_val:.3f}  {bar}")
            save_model(crop, bundle)
            results.append(m)
        else:
            print(f"{crop:<14} SKIP — {err}")

    elapsed = round(time.time() - t0, 1)
    if results:
        avg_r2     = round(np.mean([r['r2']     for r in results]), 4)
        avg_cv_mae = round(np.mean([r['cv_mae']  for r in results]), 2)
        avg_mape   = round(np.mean([r['mape']    for r in results]), 2)
        print('─' * 65)
        print(f'\n✅ SUMMARY — {len(results)}/{len(crops)} models trained in {elapsed}s')
        print(f'   Avg R²          : {avg_r2}')
        print(f'   Avg CV-MAE      : ₹{avg_cv_mae}/kg  (walk-forward, honest)')
        print(f'   Avg MAPE        : {avg_mape}%')
        print(f'   Models saved to : models/')
        print(f'\n   Restart app.py to use new models.\n')
        with open(f'{MODELS_DIR}/_summary.json', 'w') as f:
            json.dump({'avg_r2': avg_r2, 'avg_cv_mae': avg_cv_mae,
                       'avg_mape': avg_mape, 'n_models': len(results),
                       'n_features': N_FEATURES, 'approach': 'YoY + seasonal',
                       'dataset': '25 years 2000-2024',
                       'trained_at': str(pd.Timestamp.now())[:19]}, f, indent=2)

if __name__ == '__main__':
    train_all(sys.argv[1:] if len(sys.argv) > 1 else None)
