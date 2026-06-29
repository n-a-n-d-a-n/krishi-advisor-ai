# ai_engine.py — KrishiAdvisor AI
# PROPER TIME-SERIES: Uses same-month-last-year + YoY features
# NOT lag1 extrapolation — real seasonal ML prediction

import os, json, datetime, pickle
import numpy as np
import pandas as pd
from database import get_db

from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score

MODELS_DIR = os.path.join(os.path.dirname(__file__), 'models')
DATA_PATH  = os.path.join(os.path.dirname(__file__), 'data', 'maharashtra_crop_prices.csv')
N_FEATURES = 23
os.makedirs(MODELS_DIR, exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════════
#  23-FEATURE VECTOR (YoY-based — matches train_models.py exactly)
# ══════════════════════════════════════════════════════════════════════════════
def build_features_yoy(month, year, price_map, stat_info):
    """
    Build prediction feature vector using year-over-year historical data.
    price_map: dict of "year,month" -> price (full history)
    stat_info: seasonal indices, annual avgs etc from training
    """
    def gp(y, m):
        return price_map.get(f"{y},{m}")

    seas     = stat_info.get('seas', {})
    peak_mo  = stat_info.get('peak_mo', 6)
    mo_rank  = stat_info.get('mo_rank', {})
    ann_avgs = stat_info.get('ann_avgs', {})
    ann_avg  = np.mean(list(ann_avgs.values())) if ann_avgs else 20.0

    sm1 = gp(year-1, month)
    sm2 = gp(year-2, month)
    sm3 = gp(year-3, month)

    # If we don't have historical data for this month, use annual avg * seasonal factor
    seas_factor = seas.get(month, seas.get(str(month), 1.0))
    fallback    = ann_avgs.get(year-1, ann_avg) * seas_factor
    sm1 = sm1 if sm1 else fallback
    sm2 = sm2 if sm2 else fallback * 0.95
    sm3 = sm3 if sm3 else fallback * 0.90

    yoy  = (sm1 - sm2) / sm2 * 100 if sm2 > 0 else 0
    yoy2 = (sm2 - sm3) / sm3 * 100 if sm3 > 0 else 0
    sm3a = (sm1 + sm2 + sm3) / 3

    a1   = ann_avgs.get(year-1, ann_avg)
    a2   = ann_avgs.get(year-2, a1)
    a3   = ann_avgs.get(year-3, a2)
    atr  = (a1 - a3) / 2

    # Recent months momentum
    rec = []
    for b in range(1, 7):
        bm = ((month - b - 1) % 12) + 1
        by = year if month - b >= 1 else year - 1
        pp = gp(by, bm)
        if pp: rec.append(pp)
    rm3 = np.mean(rec[:3]) if len(rec) >= 3 else sm1
    rm6 = np.mean(rec[:6]) if len(rec) >= 6 else sm1
    mom = rm3 / rm6 if rm6 > 0 else 1.0

    hist = [gp(year-k, month) for k in range(1, 8)]
    hist = [p for p in hist if p]
    vol  = np.std(hist) / np.mean(hist) if len(hist) > 2 else 0.15

    si   = float(seas.get(month, seas.get(str(month), 1.0)))
    pd_  = min((peak_mo - month) % 12, (month - peak_mo) % 12)
    mr   = float(mo_rank.get(month, mo_rank.get(str(month), 6)))

    ann5 = [ann_avgs.get(year-k, a1) for k in range(1, 6) if ann_avgs.get(year-k)]
    pl   = sm1 / (np.mean(ann5) if ann5 else ann_avg)

    t    = year * 12 + month
    sin1 = np.sin(2*np.pi*month/12); cos1 = np.cos(2*np.pi*month/12)
    sin2 = np.sin(4*np.pi*month/12); cos2 = np.cos(4*np.pi*month/12)

    feat = [t,month,year,sm1,sm2,sm3,yoy,yoy2,sm3a,a1,atr,mom,rm3,rm6,vol,si,pd_,mr,sin1,cos1,sin2,cos2,pl]
    assert len(feat) == N_FEATURES
    return feat


# ══════════════════════════════════════════════════════════════════════════════
#  LOAD MODEL
# ══════════════════════════════════════════════════════════════════════════════
def load_model(crop_name, crop_id, user_prices=None):
    mf = os.path.join(MODELS_DIR, f'{crop_name}_model.pkl')
    sf = os.path.join(MODELS_DIR, f'{crop_name}_scaler.pkl')
    jf = os.path.join(MODELS_DIR, f'{crop_name}_meta.json')

    if os.path.exists(mf) and os.path.exists(sf) and os.path.exists(jf):
        try:
            with open(mf,'rb') as f: bundle = pickle.load(f)
            with open(sf,'rb') as f: scaler = pickle.load(f)
            with open(jf,'r')  as f: meta   = json.load(f)
            if meta.get('n_features', 0) != N_FEATURES:
                raise ValueError('stale model')
            return bundle, scaler, meta
        except Exception:
            for p in [mf, sf, jf]:
                if os.path.exists(p): os.remove(p)
    return None, None, None


# ══════════════════════════════════════════════════════════════════════════════
#  PRICE PREDICTION  (YoY-based, not lag1 extrapolation)
# ══════════════════════════════════════════════════════════════════════════════
SEASONAL_FALLBACK = {
    'onion':  [0.62,0.66,0.90,1.22,1.58,2.15,1.80,1.45,1.18,0.90,0.74,0.62],
    'tomato': [1.18,0.94,0.80,0.68,0.90,1.48,2.10,1.38,1.08,0.80,0.90,1.12],
    'mango':  [1.12,1.02,0.88,0.70,0.54,0.64,0.80,1.00,1.32,1.55,1.42,1.22],
    'potato': [1.04,0.97,0.91,0.86,0.91,1.03,1.14,1.24,1.20,1.11,1.04,0.98],
}


def predict_prices(crop_name, crop_id, current_price, user_prices=None, months_ahead=4):
    bundle, scaler, meta = load_model(crop_name, crop_id, user_prices)
    now = datetime.datetime.now()

    if bundle and scaler and meta:
        # Get full price map from training data
        price_map  = meta.get('full_prices', {})
        stat_info  = meta.get('stat_info', {})
        ann_avgs   = stat_info.get('ann_avgs', {})

        # Inject current price into price_map for this month
        price_map[f"{now.year},{now.month}"] = current_price

        # Inject user-provided past prices if given
        if user_prices:
            for i, p in enumerate(user_prices[-3:]):
                offset = len(user_prices) - i
                pm_    = ((now.month - offset - 1) % 12) + 1
                py_    = now.year if now.month > offset else now.year - 1
                price_map[f"{py_},{pm_}"] = float(p)

        # Also update annual avg for current year with current price
        ann_avgs[now.year] = current_price  # approximate

        preds = {}
        for m in range(1, months_ahead + 1):
            fm = ((now.month + m - 1) % 12) + 1
            fy = now.year + (now.month + m - 1) // 12

            try:
                feat = build_features_yoy(fm, fy, price_map, stat_info)
                Xp   = scaler.transform([feat])
                w    = bundle['weights']
                pred = (w[0]*bundle['gb'].predict(Xp)[0] +
                        w[1]*bundle['rf'].predict(Xp)[0] +
                        w[2]*bundle['ridge'].predict(Xp)[0])
            except Exception:
                # Fallback to seasonal adjustment from current price
                pattern = SEASONAL_FALLBACK.get(crop_name, [1.0]*12)
                cf      = pattern[now.month-1]
                pred    = current_price * (pattern[fm-1]/cf if cf>0 else 1.0) * (1+0.06*m/12)

            # For month 1: blend with current price to avoid wild jump
            if m == 1:
                pred = pred * 0.70 + current_price * 0.30

            pred = max(round(float(pred), 2), 0.5)
            preds[m] = pred

            # Add this prediction back into price_map for next iteration
            # (so month 2 knows what month 1 predicted)
            price_map[f"{fy},{fm}"] = pred

        return preds, meta

    # Full fallback: seasonal pattern
    pattern    = SEASONAL_FALLBACK.get(crop_name.lower(), [1.0]*12)
    cur_factor = pattern[now.month-1]
    preds = {}
    for m in range(1, months_ahead+1):
        fm    = ((now.month+m-1)%12)+1
        ratio = pattern[fm-1]/cur_factor if cur_factor>0 else 1.0
        preds[m] = max(round(current_price*ratio*(1+0.06*m/12), 2), 0.5)
    return preds, {}


# ══════════════════════════════════════════════════════════════════════════════
#  PROFIT ENGINE (unchanged)
# ══════════════════════════════════════════════════════════════════════════════
SPOILAGE = {0:0.0, 1:0.05, 2:0.10, 3:0.15, 4:0.20}
TEMP_ADJ = {'low':0.0, 'medium':0.02, 'high':0.05}

PROCESSING = {
    'mango':      {'yield':0.65,'labour':8,'process':7,'pack':15,'price':80,'name':'Mango Pulp / आंबा पल्प'},
    'tomato':     {'yield':0.70,'labour':5,'process':6,'pack':10,'price':45,'name':'Tomato Puree / टोमॅटो प्युरी'},
    'chilli':     {'yield':0.30,'labour':10,'process':15,'pack':20,'price':200,'name':'Chilli Powder / मिरची पावडर'},
    'turmeric':   {'yield':0.20,'labour':8,'process':12,'pack':18,'price':250,'name':'Turmeric Powder / हळद पावडर'},
    'garlic':     {'yield':0.40,'labour':10,'process':8,'pack':12,'price':150,'name':'Garlic Paste / लसूण पेस्ट'},
    'banana':     {'yield':0.80,'labour':4,'process':5,'pack':8,'price':40,'name':'Banana Chips / केळी वेफर्स'},
    'pomegranate':{'yield':0.60,'labour':10,'process':8,'pack':15,'price':120,'name':'Pomegranate Juice / डाळिंब रस'},
    'orange':     {'yield':0.65,'labour':6,'process':6,'pack':12,'price':70,'name':'Orange Juice / संत्रा रस'},
    'strawberry': {'yield':0.75,'labour':12,'process':10,'pack':20,'price':200,'name':'Strawberry Jam / जाम'},
    'guava':      {'yield':0.70,'labour':6,'process':8,'pack':12,'price':80,'name':'Guava Jelly / पेरू जेली'},
    'sugarcane':  {'yield':0.10,'labour':3,'process':5,'pack':8,'price':40,'name':'Jaggery / गूळ'},
    'coconut':    {'yield':0.50,'labour':5,'process':8,'pack':10,'price':80,'name':'Coconut Oil / खोबरेल तेल'},
    'ginger':     {'yield':0.25,'labour':8,'process':10,'pack':15,'price':180,'name':'Dry Ginger / सुंठ'},
}

def spoilage_rate(months, temp='medium'):
    return min(SPOILAGE.get(min(months,4),0.20)+TEMP_ADJ.get(temp,0.02)*months, 0.95)

def calc_profit(qty, sell_price, prod_cost, transport, storage_rate, months, temp='medium'):
    spl=spoilage_rate(months,temp); eff_qty=qty*(1-spl)
    cost=prod_cost*qty+transport*qty+storage_rate*qty*months
    rev=sell_price*eff_qty; profit=rev-cost
    bev=cost/eff_qty if eff_qty>0 else 0
    roi=profit/cost*100 if cost>0 else (100.0 if profit>0 else 0.0)
    return {'months':months,'spoilage_pct':round(spl*100,1),'effective_qty':round(eff_qty,2),
            'total_cost':round(cost,2),'revenue':round(rev,2),'profit':round(profit,2),
            'breakeven':round(bev,2),'roi_pct':round(roi,1)}

def calc_processing(crop, qty, prod_cost):
    p=PROCESSING.get(crop.lower())
    if not p: return None
    pulp_qty=qty*p['yield']
    cost=prod_cost*qty+p['labour']*qty+p['process']*qty+p['pack']*pulp_qty
    rev=p['price']*pulp_qty; profit=rev-cost
    return {'strategy':f"प्रक्रिया → {p['name']}",'strategy_type':'process',
            'months':0,'spoilage_pct':0,'effective_qty':round(pulp_qty,2),
            'total_cost':round(cost,2),'revenue':round(rev,2),'profit':round(profit,2),
            'breakeven':round(cost/pulp_qty if pulp_qty>0 else 0,2),
            'roi_pct':round(profit/cost*100 if cost>0 else 0,1),
            'product_name':p['name'],'predicted_price':p['price']}

def simulate_all(crop_name,qty,current_price,prod_cost,pred_prices,transport,storage_rate,temperature):
    strategies=[]
    r=calc_profit(qty,current_price,prod_cost,0,0,0,temperature)
    r.update({'strategy':'आत्ता विका / Sell Now','strategy_type':'sell','predicted_price':current_price})
    strategies.append(r)
    for m in range(1,5):
        sp=pred_prices.get(m,current_price)
        r=calc_profit(qty,sp,prod_cost,transport,storage_rate,m,temperature)
        r.update({'strategy':f'{m} महिने साठवा / Store {m}M','strategy_type':'store','predicted_price':sp})
        strategies.append(r)
    best_m=max(range(1,5),key=lambda m:pred_prices.get(m,0))
    best_sp=pred_prices.get(best_m,current_price); half=qty/2
    r1=calc_profit(half,current_price,prod_cost,0,0,0,temperature)
    r2=calc_profit(half,best_sp,prod_cost,transport,storage_rate,best_m,temperature)
    pc=r1['total_cost']+r2['total_cost']; pe=half+r2['effective_qty']
    strategies.append({'strategy':f'अर्धे-अर्धे / Partial (50%+{best_m}M)','strategy_type':'partial',
        'months':best_m,'spoilage_pct':r2['spoilage_pct'],'effective_qty':round(pe,2),
        'total_cost':round(pc,2),'revenue':round(r1['revenue']+r2['revenue'],2),
        'profit':round(r1['profit']+r2['profit'],2),'breakeven':round(pc/pe if pe>0 else 0,2),
        'roi_pct':round((r1['profit']+r2['profit'])/pc*100 if pc>0 else 0,1),'predicted_price':best_sp})
    proc=calc_processing(crop_name,qty,prod_cost)
    if proc: strategies.append(proc)
    strategies.sort(key=lambda x:x['profit'],reverse=True)
    return strategies

def assess_risk(s,current_price,mae=None):
    profit=s.get('profit',0); stype=s.get('strategy_type','sell')
    months=s.get('months',0); sp=s.get('spoilage_pct',0)
    pred_p=s.get('predicted_price',current_price); roi=s.get('roi_pct',0)
    acc=f' (CV-MAE ±₹{mae:.1f}/kg)' if mae else ''
    if profit<0:
        return 'High',f'⚠️ नुकसान ₹{abs(profit):.0f} होईल! / Loss of ₹{abs(profit):.0f} — not recommended!'
    if stype=='sell':
        return 'Low',f'✅ तात्काळ विक्री — धोका नाही. ROI: {roi:.1f}%{acc}'
    if stype=='process':
        return 'Low',f'🏭 प्रक्रिया — खराबी नाही, जास्त मूल्य. ROI: {roi:.1f}%'
    gain=(pred_p-current_price)/current_price*100 if current_price>0 else 0
    if months<=1: return 'Low',f'📦 {months} महिना — कमी धोका. भाव: ₹{pred_p} (+{gain:.1f}%){acc}'
    elif months<=2: return 'Medium',f'⚖️ {months} महिने — मध्यम धोका. खराबी: {sp}%, भाव: ₹{pred_p}{acc}'
    return 'High',f'⚠️ {months} महिने — जास्त खराबी ({sp}%). शीतगृह आवश्यक.{acc}'


def full_analysis(crop_id,crop_name,qty,current_price,prod_cost,transport,storage_rate,temperature,user_prices=None):
    pred_prices,meta=predict_prices(crop_name,crop_id,current_price,user_prices,months_ahead=4)
    mae=meta.get('cv_mae') or meta.get('mae')
    strategies=simulate_all(crop_name,qty,current_price,prod_cost,pred_prices,transport,storage_rate,temperature)
    best=strategies[0]; risk,reason=assess_risk(best,current_price,mae)
    comparison=[]
    for s in strategies:
        rl,_=assess_risk(s,current_price,None)
        comparison.append({'strategy':s['strategy'],'profit':s['profit'],'revenue':s['revenue'],
                           'total_cost':s['total_cost'],'roi_pct':s.get('roi_pct',0),
                           'spoilage':s.get('spoilage_pct',0),'risk':rl,'is_best':s['strategy']==best['strategy']})
    return {
        'crop':crop_name,'best_strategy':best['strategy'],
        'expected_profit':best['profit'],'revenue':best['revenue'],
        'total_cost':best['total_cost'],'breakeven':best['breakeven'],
        'roi_pct':best.get('roi_pct',0),'risk':risk,'reason':reason,
        'spoilage_pct':best.get('spoilage_pct',0),
        'predicted_prices':pred_prices,'all_strategies':comparison,
        'model_accuracy':{
            'mae':mae,'r2':meta.get('r2'),'mape':meta.get('mape'),
            'cv_mae':meta.get('cv_mae'),
            'dataset':'25 years (2000–2024) · YoY features',
            'n_features':N_FEATURES,'model':'Ensemble GB+RF+Ridge · YoY seasonal',
            'note':f"R²={meta.get('r2')} | CV-MAE=₹{mae}/kg | MAPE={meta.get('mape')}%" if mae else 'Seasonal fallback'
        }
    }

def pretrain_all_models():
    pass  # Models already trained and saved
