# 📊 Unemployment Rate Forecasting

**World Bank Data 2010–2024 | 187 Countries | Regression Analysis**

## 🚀 Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Run Streamlit App
```bash
streamlit run app.py
```

### 3. Open Jupyter Notebook
```bash
jupyter notebook unemployment_forecasting.ipynb
```

## 📁 Project Files
| File | Description |
|------|-------------|
| `app.py` | Streamlit interactive web app |
| `unemployment_forecasting.ipynb` | Jupyter notebook with step-by-step analysis |
| `world_bank_data_2025.csv` | World Bank dataset |
| `requirements.txt` | Python dependencies |

## 🔧 Features
- Select any of **187 countries**
- **5 regression models**: Linear, Polynomial (deg 2 & 3), Exponential, Moving Average
- Forecast up to **any year** (2025–2034) — including **2027**
- 95% confidence intervals
- Residual analysis plots
- All-models comparison table
- Export forecast data

## 📌 Key Fix
The forecast horizon slider now lets you pick **any target year** (e.g. 2027),  
not just a fixed number of years forward.
