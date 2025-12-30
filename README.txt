# Sample Disturbance Prediction App

A **Streamlit web application** to predict **sampling disturbance (Δe/e₀) in soils** using a trained **XGBoost model**.  
The app also calculates the **Area Ratio (AR)** from sampler dimensions and classifies the **sample quality** based on **Overconsolidation Ratio (OCR)**.  

It is designed for engineers and geotechnical researchers to quickly assess soil sample disturbance and quality using input parameters.

---

## Features

- **Area Ratio Calculator:** Automatically calculate AR from sampler external diameter and wall thickness.  
- **Dynamic Schematic:** Visual diagram of sampler cross-section that updates based on dimensions.  
- **Prediction of Δe/e₀:** Uses AR, Plasticity Index (PI), Effective Normal Stress, and OCR to predict sampling disturbance.  
- **Sample Quality Categories:** Classifies samples (Excellent, Good, Poor, Very Poor) based on OCR and predicted Δe/e₀.  
- **Prediction History:** Keeps a session-based history of all predictions with download option as CSV.  

---

## Requirements

- Python 3.9+
- Streamlit
- joblib
- numpy
- pandas
- matplotlib
- xgboost

Install dependencies using:

```bash
pip install streamlit joblib numpy pandas matplotlib xgboost
