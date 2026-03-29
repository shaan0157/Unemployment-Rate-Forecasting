# 📊 Unemployment Rate Forecasting App

A Machine Learning-based web application that predicts future unemployment rates using historical data from the World Bank.

🔗 **Live App:** https://unemployment-rate-forecasting-hpz94dfkkhhb4gfzt4wzif.streamlit.app/

---

## 🌍 Project Overview

This project uses regression models to analyze historical unemployment data and forecast future trends up to the year 2100. It provides interactive visualizations and model comparisons to help understand economic patterns.

---

## 🚀 Features

* 📌 Country-wise unemployment analysis (187 countries)
* 📈 Multiple regression models:

  * Linear Regression
  * Polynomial Regression (Degree 2 & 3)
  * Exponential Model
  * Moving Average (3-year)
* 🔮 Future forecasting up to 2100
* 📊 Interactive graphs with confidence intervals
* 📋 Forecast data table with yearly changes
* 📉 Residual analysis for model evaluation
* 🔬 Model comparison using R², RMSE, MAE

---

## 🛠️ Technologies Used

* Python
* Streamlit
* Pandas
* NumPy
* Matplotlib
* Scikit-learn

---

## 📂 Project Structure

```
unemployment-forecasting-app/
│
├── app.py
├── full_dataset_2000_2100.csv
├── requirements.txt
├── README.md
└── image.png (optional)
```

---

## ▶️ How to Run Locally

1. Clone the repository:

```
git clone https://github.com/shaan0157/unemployment-forecasting-app.git
```

2. Navigate to the project folder:

```
cd unemployment-forecasting-app
```

3. Install dependencies:

```
pip install -r requirements.txt
```

4. Run the app:

```
streamlit run app.py
```

---

## 📊 Dataset

* Source: World Bank Data
* Covers unemployment rates across multiple countries
* Time range: 2000 – 2100 (extended dataset)

---

## 📈 Model Evaluation Metrics

* **R² Score** – Model accuracy
* **RMSE** – Error measurement
* **MAE** – Average prediction error

---

## 🎯 Use Cases

* Economic trend analysis
* Policy planning insights
* Academic and research purposes
* Data science project demonstration

---

## 📌 Future Improvements

* Add advanced ML models (ARIMA, Random Forest)
* Real-time data integration
* Improved UI/UX dashboard
* Multi-country comparison

---

## 👨‍💻 Author

**Shantanu Apurva**
Engineering Student | Data Science Enthusiast

---

## ⭐ Acknowledgements

* World Bank for dataset
* Streamlit for deployment platform

---

## 📢 Note

This project is developed for educational purposes and may not reflect real-world economic predictions with complete accuracy.
