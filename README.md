# 🌾 KrishiAdvisor AI — कृषी सल्लागार

> **AI-powered crop price prediction & farm profit optimization for Maharashtra farmers**
> Built as a Hackathon Prototype · Team of 4 · March 2026

[![Live Demo](https://img.shields.io/badge/Live%20Demo-krishi--advisor--ai.onrender.com-brightgreen?style=flat&logo=render)](https://krishi-advisor-ai.onrender.com)
[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat&logo=python&logoColor=white)](https://python.org)
[![Flask](https://img.shields.io/badge/Flask-3.1.3-000000?style=flat&logo=flask)](https://flask.palletsprojects.com)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.8.0-F7931E?style=flat&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![SQLite](https://img.shields.io/badge/SQLite-3-003B57?style=flat&logo=sqlite)](https://sqlite.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## 🌐 Live Demo

**[https://krishi-advisor-ai.onrender.com](https://krishi-advisor-ai.onrender.com)**

> Note: Hosted on Render free tier — may take 30–60 seconds to wake up on first visit.

---

## 📌 Problem Statement

Indian farmers lose **30–40% of potential income** due to poor timing of crop sales and lack of market intelligence. They make sell/store decisions based on hearsay, not data — with no bilingual tool accessible to Marathi-speaking farmers.

**KrishiAdvisor AI solves this** by predicting crop prices up to 4 months ahead and recommending the single best strategy — Sell Now, Store, or Process — to maximise farm profit.

---

## ✨ Features

- 🤖 **AI Price Prediction** — 30 crop-specific ensemble ML models (Gradient Boosting + Random Forest + Ridge) trained on 25 years of Maharashtra price data (2000–2024)
- 📊 **Strategy Comparison** — Ranks Sell Now / Store 1–4 months / Partial / Process strategies by profit, ROI, and risk
- 🦠 **Crop-Specific Spoilage Model** — Real post-harvest spoilage rates per crop (ICAR/NHB data), compound decay, and storage-type awareness (home / warehouse / cold chain)
- 🌐 **Bilingual UI** — Full Marathi + English interface throughout
- 🧑‍🌾 **Farm Management** — Stock tracking, sales history, profit dashboard
- 🔌 **REST API** — `/api/predict` endpoint for integration with mobile apps or WhatsApp bots

---

## 🧠 ML Model Architecture

Each of the **30 crop-specific models** is a weighted ensemble:

```
Final Price = 0.55 × GradientBoosting + 0.35 × RandomForest + 0.10 × Ridge
```

### Key Innovation — Year-over-Year (YoY) Feature Engineering

Unlike naive models that use last month's price (lag-1), KrishiAdvisor uses **same-month-last-year** as its primary predictor — because June's price this year is best predicted from June last year, not from May this year.

**23 features per sample:**
- Same-month prices for last 1, 2, 3 years (`sm1`, `sm2`, `sm3`)
- Year-over-year % change (`yoy_chg`, `yoy2_chg`)
- Seasonal index, peak-month distance, month rank
- sin/cos harmonic seasonality (4 features)
- Annual trend, rolling momentum (3M vs 6M), volatility index
- Normalised price level vs 5-year average

### Model Performance (30 crops)

| Metric | Min | Average | Max |
|--------|-----|---------|-----|
| R² Score | 0.771 | **0.955** | 0.990 |
| MAPE | 2.02% | **6.64%** | 27.55% |
| Training data span | 25 years | 25 years | 25 years |

<details>
<summary>📋 Per-crop accuracy (click to expand)</summary>

| Crop | R² | MAPE |
|------|----|------|
| Cotton | 0.9895 | 2.94% |
| Wheat | 0.9855 | 2.02% |
| Sugarcane | 0.9855 | 2.35% |
| Pomegranate | 0.9828 | 3.85% |
| Turmeric | 0.9821 | 3.39% |
| Coconut | 0.9801 | 3.12% |
| Soybean | 0.9763 | 3.20% |
| Orange | 0.9751 | 4.69% |
| Grapes | 0.9748 | 4.87% |
| Banana | 0.9709 | 3.53% |
| Mango | 0.9693 | 6.21% |
| Onion | 0.9358 | 13.79% |
| Tomato | 0.7709 | 27.55%* |

*Tomato is India's most volatile crop — swings ₹5 to ₹160/kg within weeks.

</details>

---

## 🦠 Crop-Specific Spoilage Model

Spoilage varies by crop biology, temperature, and storage type:

| Crop | Category | 1M (warehouse) | 3M (warehouse) | 3M (cold) |
|------|----------|----------------|----------------|-----------|
| Strawberry | Very Perishable | 50.0% | 87.5% | 16.9% |
| Tomato | Very Perishable | 30.0% | 65.7% | 20.8% |
| Mango | Perishable | 28.0% | 62.7% | 15.9% |
| Onion | Storable | 6.0% | 16.9% | 7.0% |
| Potato | Storable | 5.0% | 14.3% | 4.4% |
| Wheat | Durable | 1.0% | 3.0% | 1.5% |

Uses **compound decay**: `spoilage = 1 − (1 − monthly_rate)^months`

Storage types: `home` · `warehouse` · `cold`

---

## 🌱 Crops Covered (30 Total)

| Category | Crops |
|----------|-------|
| Vegetables | Onion, Tomato, Potato, Cauliflower, Cabbage, Brinjal, Okra, Peas, Carrot, Radish, Spinach |
| Fruits | Mango, Grapes, Banana, Orange, Pomegranate, Guava, Lemon, Coconut, Strawberry, Papaya, Watermelon |
| Spices | Chilli, Garlic, Ginger, Turmeric |
| Grains | Wheat |
| Cash Crops | Soybean, Cotton, Sugarcane |

---

## 🗂️ Project Structure

```
krishi-advisor-ai/
├── app.py
├── ai_engine.py
├── train_models.py
├── build_dataset.py
├── database.py
├── data_fetcher.py
├── fetch_api_data.py
├── requirements.txt
├── render.yaml
├── vercel.json
├── README.md
├── KrishiAdvisor_Hackathon_Documentation.docx
├── models/
│   ├── banana_model.pkl
│   ├── banana_scaler.pkl
│   ├── banana_meta.json
│   ├── ... (27 more crops)
│   ├── _summary.json
│   └── training_summary.json
├── data/
│   ├── krishi.db
│   ├── maharashtra_crop_prices.csv
│   └── large_dataset.csv
├── templates/
│   ├── base.html
│   ├── home.html
│   ├── dashboard.html
│   ├── advisor.html
│   ├── stock.html
│   ├── sales.html
│   ├── login.html
│   ├── register.html
│   ├── profile.html
│   └── about.html
└── uploads/
```

---

## 🚀 Getting Started

```bash
# 1. Clone the repo
git clone https://github.com/n-a-n-d-a-n/krishi-advisor-ai.git
cd krishi-advisor-ai

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the app
python app.py
```

Open `http://localhost:5000`

> Pre-trained models are included — no need to retrain.

---

## 🔌 API

### `POST /api/predict`
```json
{
  "crop_name": "onion",
  "quantity": 500,
  "current_price": 12.5,
  "production_cost": 6.0,
  "transport_cost": 2.0,
  "storage_cost": 1.5,
  "temperature": "medium",
  "storage_type": "warehouse"
}
```

### `GET /api/crops`
Returns all 30 supported crops with Marathi names.

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| Backend | Flask 3.1.3 |
| ML | scikit-learn 1.8.0 (GBR, RFR, Ridge, TimeSeriesSplit) |
| Data | Pandas 3.0.1, NumPy 2.4.3 |
| Database | SQLite 3 |
| Frontend | Jinja2 + Bootstrap (Marathi + English) |
| Deployment | Render |

---

## 🗺️ Roadmap

- [ ] Live AGMARKNET API integration
- [ ] Weather feature integration (IMD monsoon data)
- [ ] WhatsApp chatbot interface
- [ ] Farmer-to-buyer marketplace

---

## 👥 Team

Built by team Harvest Helions of 4 as a Hackathon prototype — Maharashtra, India · March 2026

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

---

## 🙏 Data Sources

- [AGMARKNET](https://agmarknet.gov.in) — Maharashtra APMC market price patterns
- [NHB Horticulture Statistics](https://nhb.gov.in) — baseline data
- [ICAR CIPHET](https://ciphet.in) — post-harvest spoilage benchmarks
- [IMD](https://imd.gov.in) — monsoon seasonal patterns
