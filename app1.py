"""
Unemployment Rate Forecasting App
World Bank Data 2010–2024 | 187 Countries
Run with: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# ── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Unemployment Forecasting",
    page_icon="📊",
    layout="wide"
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main { padding-top: 1rem; }
    .metric-card {
        background: #f8f9fa;
        border-radius: 10px;
        padding: 14px 18px;
        border-left: 4px solid #2196F3;
    }
    .stSelectbox label, .stSlider label, .stRadio label { font-weight: 600; }
    h1 { color: #1a1a2e; }
    h2 { color: #16213e; }
    h3 { color: #0f3460; }
</style>
""", unsafe_allow_html=True)

# ── Load Data ─────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv("world_bank_data_2025.csv")
    df = df[['country_name', 'year', 'Unemployment Rate (%)']].copy()
    df.columns = ['country', 'year', 'unemployment']
    df = df.dropna(subset=['unemployment'])
    df['year'] = df['year'].astype(int)
    df['unemployment'] = df['unemployment'].astype(float)
    return df

df = load_data()
countries = sorted(df['country'].unique().tolist())

# ── Regression Functions ───────────────────────────────────────────────────────
def fit_model(X, y, model_type, degree=2):
    X2d = X.reshape(-1, 1)
    if model_type == "Linear":
        model = LinearRegression()
        model.fit(X2d, y)
        fitted = model.predict(X2d)
        a = model.intercept_
        b = model.coef_[0]
        eq = f"y = {a:.3f} + {b:.4f}·x"
        return model, fitted, eq
    elif model_type == "Polynomial (degree 2)":
        model = make_pipeline(PolynomialFeatures(2), LinearRegression())
        model.fit(X2d, y)
        fitted = model.predict(X2d)
        coef = model.named_steps['linearregression'].coef_
        intercept = model.named_steps['linearregression'].intercept_
        eq = f"y = {intercept:.3f} + {coef[1]:.4f}·x + {coef[2]:.6f}·x²"
        return model, fitted, eq
    elif model_type == "Polynomial (degree 3)":
        model = make_pipeline(PolynomialFeatures(3), LinearRegression())
        model.fit(X2d, y)
        fitted = model.predict(X2d)
        coef = model.named_steps['linearregression'].coef_
        intercept = model.named_steps['linearregression'].intercept_
        eq = f"y = {intercept:.3f} + {coef[1]:.4f}·x + {coef[2]:.5f}·x² + {coef[3]:.7f}·x³"
        return model, fitted, eq
    elif model_type == "Exponential":
        y_safe = np.where(y > 0, y, 0.01)
        log_y = np.log(y_safe)
        lr = LinearRegression()
        lr.fit(X2d, log_y)
        a = np.exp(lr.intercept_)
        b = lr.coef_[0]
        class ExpModel:
            def predict(self, X): return a * np.exp(b * X.ravel())
        model = ExpModel()
        fitted = model.predict(X2d)
        eq = f"y = {a:.3f} · e^({b:.5f}·x)"
        return model, fitted, eq
    elif model_type == "Moving Average (3yr)":
        series = pd.Series(y)
        ma = series.rolling(3, min_periods=1).mean().values
        # Use linear trend from last 5 points for forecasting
        last_n = min(5, len(X))
        trend_model = LinearRegression()
        trend_model.fit(X[-last_n:].reshape(-1, 1), y[-last_n:])
        eq = f"3-year MA · Trend slope = {trend_model.coef_[0]:.4f}/yr"
        class MAModel:
            def predict(self, X_new): return trend_model.predict(X_new.reshape(-1, 1))
        return MAModel(), ma, eq

def make_forecast(model, last_year, forecast_up_to_year, rmse):
    future_years = np.arange(last_year + 1, forecast_up_to_year + 1)
    if len(future_years) == 0:
        return future_years, np.array([]), np.array([]), np.array([])
    preds = model.predict(future_years.reshape(-1, 1))
    preds = np.maximum(preds, 0)
    # Expanding confidence interval
    ci_width = rmse * np.array([1.5 + i * 0.3 for i in range(len(future_years))])
    lower = np.maximum(0, preds - ci_width * 1.96)
    upper = preds + ci_width * 1.96
    return future_years, preds, lower, upper

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/8/8c/World_Bank_logo.svg/200px-World_Bank_logo.svg.png", width=140)
    st.markdown("## 🌍 Unemployment Forecaster")
    st.markdown("**World Bank Data 2010–2024**")
    st.markdown("---")

    # Country selector
    default_idx = countries.index("India") if "India" in countries else 0
    country = st.selectbox("🗺 Select Country", countries, index=default_idx)

    # Model selector
    model_type = st.radio(
        "📐 Regression Model",
        ["Linear", "Polynomial (degree 2)", "Polynomial (degree 3)",
         "Exponential", "Moving Average (3yr)"],
        index=0
    )

    # Forecast up to year
    st.markdown("### 📅 Forecast Target Year")
    max_data_year = int(df['year'].max())
    forecast_target = st.number_input(
    "Enter forecast year",
    min_value=max_data_year + 1,
    max_value=2100,
    value=max_data_year + 5,
    step=1
    )

    # Show model info
    st.markdown("---")
    st.markdown("### ℹ️ Model Info")
    model_info = {
        "Linear": "Simple straight-line trend. Best for stable economies.",
        "Polynomial (degree 2)": "Captures curves & turning points (e.g. post-crisis recovery).",
        "Polynomial (degree 3)": "Fits complex S-shaped patterns but risks overfitting.",
        "Exponential": "For accelerating growth or decline patterns.",
        "Moving Average (3yr)": "Smooths noise; uses recent slope for forecasting."
    }
    st.info(model_info[model_type])

# ── Main Content ───────────────────────────────────────────────────────────────
st.title("📊 Unemployment Rate Forecasting")
st.markdown(f"**Country:** `{country}` | **Model:** `{model_type}` | **Forecast to:** `{forecast_target}`")
st.markdown("---")

# Filter country data
cdata = df[df['country'] == country].sort_values('year')
if len(cdata) < 4:
    st.error(f"Not enough data for {country}. Need at least 4 data points.")
    st.stop()

X = cdata['year'].values.astype(float)
y = cdata['unemployment'].values.astype(float)
last_year = int(X.max())
n_forecast_years = forecast_target - last_year

# Fit model
model, fitted, equation = fit_model(X, y, model_type)

# Metrics
r2 = r2_score(y, fitted)
rmse = np.sqrt(mean_squared_error(y, fitted))
mae = mean_absolute_error(y, fitted)

# Forecast
future_years, forecast_vals, lower, upper = make_forecast(model, last_year, forecast_target, rmse)

# ── KPI Cards ──────────────────────────────────────────────────────────────────
col1, col2, col3, col4, col5 = st.columns(5)
with col1:
    color = "🟢" if r2 > 0.8 else "🟡" if r2 > 0.5 else "🔴"
    st.metric(f"{color} R² Score", f"{r2:.4f}")
with col2:
    st.metric("📉 RMSE", f"{rmse:.3f}%")
with col3:
    st.metric("📐 MAE", f"{mae:.3f}%")
with col4:
    st.metric("📋 Data Points", len(y))
with col5:
    st.metric("🔮 Forecast Years", n_forecast_years)

st.markdown("---")

# ── Equation Box ───────────────────────────────────────────────────────────────
st.markdown(f"**Model Equation:** `{equation}`")

# ── Main Chart ─────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(13, 5.5))
fig.patch.set_facecolor('#fafafa')
ax.set_facecolor('#fafafa')

# Plot actual data
ax.plot(X, y, 'o-', color='#2196F3', linewidth=2, markersize=5,
        label='Actual Data', zorder=4)

# Plot fitted line
ax.plot(X, fitted, '--', color='#4CAF50', linewidth=2,
        label='Fitted Model', zorder=3)

# Plot forecast
if len(future_years) > 0:
    # Connect last actual to first forecast
    connect_x = [last_year, future_years[0]]
    connect_y = [y[-1], forecast_vals[0]]
    ax.plot(connect_x, connect_y, '--', color='#FF5722', linewidth=2, alpha=0.5)

    ax.plot(future_years, forecast_vals, 'D-', color='#FF5722', linewidth=2.5,
            markersize=7, label=f'Forecast ({last_year+1}–{forecast_target})', zorder=5)

    ax.fill_between(future_years, lower, upper, color='#FF5722', alpha=0.12,
                    label='95% Confidence Band')

    # Annotate forecast values
    for fy, fv in zip(future_years, forecast_vals):
        ax.annotate(f'{fv:.2f}%',
                    xy=(fy, fv),
                    xytext=(0, 12),
                    textcoords='offset points',
                    ha='center', fontsize=8.5,
                    color='#FF5722', fontweight='bold')

# Vertical divider
ax.axvline(x=last_year + 0.5, color='gray', linestyle=':', alpha=0.6, linewidth=1.2)
ax.text(last_year + 0.55, ax.get_ylim()[1] * 0.95 if ax.get_ylim()[1] > 0 else 1,
        '← Historical | Forecast →', fontsize=8, color='gray', va='top')

ax.set_xlabel('Year', fontsize=12)
ax.set_ylabel('Unemployment Rate (%)', fontsize=12)
ax.set_title(f'{country} — Unemployment Rate Forecast using {model_type}',
             fontsize=14, fontweight='bold', pad=12)
ax.legend(loc='upper right', framealpha=0.9, fontsize=9)
ax.grid(True, alpha=0.25, linestyle='--')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)


if forecast_target in list(future_years):
    idx = list(future_years).index(forecast_target)
    ax.axvline(x=forecast_target, color='purple', linestyle='-.', alpha=0.5, linewidth=1.2)
    ax.annotate(f'{forecast_target}: {forecast_vals[idx]:.2f}%',
                xy=(forecast_target, forecast_vals[idx]),
                xytext=(-50, -25),
                textcoords='offset points',
                arrowprops=dict(arrowstyle='->', color='purple'),
                fontsize=9, color='purple', fontweight='bold')

st.pyplot(fig)
plt.close()

# ── Forecast Table ─────────────────────────────────────────────────────────────
if len(future_years) > 0:
    st.markdown("### 🔮 Forecast Details")
    forecast_df = pd.DataFrame({
        'Year': future_years.astype(int),
        'Predicted Unemployment (%)': np.round(forecast_vals, 3),
        'Lower Bound 95% CI (%)': np.round(lower, 3),
        'Upper Bound 95% CI (%)': np.round(upper, 3),
        'YoY Change (%)': np.round(
            np.concatenate([[forecast_vals[0] - y[-1]], np.diff(forecast_vals)]), 3
        )
    })
    forecast_df['Trend'] = forecast_df['YoY Change (%)'].apply(
        lambda x: '📈 Rising' if x > 0.2 else ('📉 Falling' if x < -0.2 else '➡️ Stable')
    )

    # Style dataframe
    def highlight_change(val):
        if isinstance(val, float):
            if val > 0.2: return 'color: #E53935; font-weight: bold'
            elif val < -0.2: return 'color: #43A047; font-weight: bold'
        return ''

    styled = forecast_df.style.applymap(highlight_change, subset=['YoY Change (%)'])
    st.dataframe(styled, use_container_width=True, hide_index=True)

# ── Historical Data Table ──────────────────────────────────────────────────────
with st.expander("📋 View Full Historical Data"):
    hist_df = cdata[['year', 'unemployment']].copy()
    hist_df.columns = ['Year', 'Unemployment Rate (%)']
    hist_df['Fitted Value (%)'] = np.round(fitted, 3)
    hist_df['Residual (%)'] = np.round(y - fitted, 3)
    st.dataframe(hist_df, use_container_width=True, hide_index=True)

# ── Residual Plot ──────────────────────────────────────────────────────────────
with st.expander("📊 Residual Analysis"):
    col_a, col_b = st.columns(2)
    with col_a:
        fig2, ax2 = plt.subplots(figsize=(6, 4))
        residuals = y - fitted
        ax2.bar(X, residuals, color=['#E53935' if r > 0 else '#43A047' for r in residuals],
                alpha=0.75, edgecolor='white')
        ax2.axhline(0, color='black', linewidth=0.8)
        ax2.set_title('Residuals Over Time', fontweight='bold')
        ax2.set_xlabel('Year'); ax2.set_ylabel('Residual (%)')
        ax2.grid(True, alpha=0.2); ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)
        st.pyplot(fig2); plt.close()
    with col_b:
        fig3, ax3 = plt.subplots(figsize=(6, 4))
        ax3.scatter(fitted, residuals, color='#2196F3', alpha=0.7, s=60, edgecolors='white')
        ax3.axhline(0, color='red', linestyle='--', linewidth=1)
        ax3.set_title('Residuals vs Fitted Values', fontweight='bold')
        ax3.set_xlabel('Fitted Values (%)'); ax3.set_ylabel('Residuals (%)')
        ax3.grid(True, alpha=0.2); ax3.spines['top'].set_visible(False)
        ax3.spines['right'].set_visible(False)
        st.pyplot(fig3); plt.close()

# ── Model Comparison ───────────────────────────────────────────────────────────
with st.expander("🔬 Compare All Models"):
    all_models = ["Linear", "Polynomial (degree 2)", "Polynomial (degree 3)",
                  "Exponential", "Moving Average (3yr)"]
    comparison_rows = []
    for m in all_models:
        try:
            _, f_vals, _ = fit_model(X, y, m)
            r2_m = r2_score(y, f_vals)
            rmse_m = np.sqrt(mean_squared_error(y, f_vals))
            mae_m = mean_absolute_error(y, f_vals)
            _, fc_vals, _, _ = make_forecast(
                fit_model(X, y, m)[0], last_year, forecast_target, rmse_m
            )
            pred_target = fc_vals[-1] if len(fc_vals) > 0 else "N/A"
            comparison_rows.append({
            'Model': m,
            'R²': round(r2_m, 4),
            'RMSE (%)': round(rmse_m, 3),
            'MAE (%)': round(mae_m, 3),
            f'Predicted {forecast_target} (%)': round(fc_vals[-1], 2) if len(fc_vals) > 0 else "N/A"
            })
        except Exception:
            pass

    comp_df = pd.DataFrame(comparison_rows)
    best_r2 = comp_df['R²'].idxmax()

    def highlight_best(row):
        if row.name == best_r2:
            return ['background-color: #e8f5e9'] * len(row)
        return [''] * len(row)

    st.markdown(f"🏆 **Best model by R²:** `{comp_df.loc[best_r2, 'Model']}`")
    st.dataframe(comp_df.style.apply(highlight_best, axis=1),
                 use_container_width=True, hide_index=True)

# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<div style='text-align:center;color:gray;font-size:12px'>"
    "Data Source: World Bank 2025 · 187 Countries · 2010–2024 · "
    "Built with Python + Streamlit + scikit-learn"
    "</div>", unsafe_allow_html=True
)
