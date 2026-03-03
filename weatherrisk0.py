#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  2 13:16:34 2026

@author: sagarkumar
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Kericho Multi-Hazard Climate Risk Model
Author: Sagar Kumar

Features:
Stepwise workflow
Hailstorm detection
Climate Risk Index (CRI)
Crop Loss Estimation
Insurance Premium
Machine Learning Risk Prediction
"""

# =====================================================
# STEP 1: IMPORT LIBRARIES
# =====================================================

import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

print("\nSTEP 1: Libraries Loaded")

# =====================================================
# STEP 2: DOWNLOAD WEATHER DATA (NASA POWER)
# =====================================================

print("\nSTEP 2: Downloading Weather Data...")

latitude = -0.367
longitude = 35.283
end_date = datetime.now().strftime("%Y%m%d")

url = (
    "https://power.larc.nasa.gov/api/temporal/daily/point?"
    "start=20100101"
    f"&end={end_date}"
    f"&latitude={latitude}"
    f"&longitude={longitude}"
    "&community=ag"
    "&parameters=T2M,T2M_MAX,T2M_MIN,PRECTOTCORR,WS2M,RH2M"
    "&format=JSON"
)

response = requests.get(url, timeout=30)
response.raise_for_status()

data = response.json()
params = data['properties']['parameter']

weather_df = pd.DataFrame({
    'date': pd.to_datetime(list(params['T2M'].keys())),
    'temp_avg': list(params['T2M'].values()),
    'temp_max': list(params['T2M_MAX'].values()),
    'temp_min': list(params['T2M_MIN'].values()),
    'precip': list(params['PRECTOTCORR'].values()),
    'wind': list(params['WS2M'].values()),
    'humidity': list(params['RH2M'].values())
})

weather_df.to_csv("step1_raw_weather.csv", index=False)
print("Weather data saved")

# =====================================================
# STEP 3: DATA CLEANING
# =====================================================

print("\nSTEP 3: Cleaning Data")

weather_df.replace(-999, np.nan, inplace=True)
weather_df.ffill(inplace=True)
weather_df.bfill(inplace=True)

weather_df['year'] = weather_df['date'].dt.year
weather_df['month'] = weather_df['date'].dt.month

weather_df.to_csv("step2_clean_weather.csv", index=False)

# =====================================================
# STEP 4: EXTREME WEATHER INDICATORS
# =====================================================

print("\nSTEP 4: Creating Hazard Indicators")

weather_df['heavy_rain'] = (weather_df['precip'] > 50).astype(int)
weather_df['dry_day'] = (weather_df['precip'] < 1).astype(int)
weather_df['heat_stress'] = (weather_df['temp_max'] > 32).astype(int)
weather_df['high_wind'] = (weather_df['wind'] > 6).astype(int)

weather_df['dry_spell'] = (
    weather_df['dry_day']
    .rolling(7)
    .sum()
    .fillna(0)
    .apply(lambda x: 1 if x >= 7 else 0)
)

weather_df.to_csv("step3_hazards.csv", index=False)

# =====================================================
# STEP 5: HAILSTORM DETECTION (PROXY)
# =====================================================

print("\nSTEP 5: Detecting Hail Events")

weather_df['hail_cond1'] = (weather_df['precip'] > 20)
weather_df['hail_cond2'] = (weather_df['temp_min'] < 15)
weather_df['hail_cond3'] = (weather_df['wind'] > 6)
weather_df['hail_cond4'] = (weather_df['humidity'] > 80)

weather_df['hail_score'] = weather_df[
    ['hail_cond1','hail_cond2','hail_cond3','hail_cond4']
].sum(axis=1)

weather_df['hail_event'] = (weather_df['hail_score'] >= 3).astype(int)

weather_df.to_csv("step4_hail_events.csv", index=False)

# =====================================================
# STEP 6: CLIMATE RISK INDEX (CRI)
# =====================================================

print("\nSTEP 6: Calculating Climate Risk Index")

weather_df['CRI_daily'] = (
    weather_df['dry_spell'] * 0.30 +
    weather_df['heavy_rain'] * 0.25 +
    weather_df['heat_stress'] * 0.15 +
    weather_df['high_wind'] * 0.10 +
    weather_df['hail_event'] * 0.20
)

annual_cri = weather_df.groupby('year')['CRI_daily'].sum().reset_index()

annual_cri['CRI'] = (
    annual_cri['CRI_daily'] /
    annual_cri['CRI_daily'].max()
) * 100

annual_cri.to_csv("step5_annual_cri.csv", index=False)

# =====================================================
# STEP 7: CROP LOSS & ECONOMIC IMPACT
# =====================================================

print("\nSTEP 7: Estimating Crop Loss")

farm_area = 50
value_per_ha = 1800

annual_cri['loss_percent'] = annual_cri['CRI'] * 0.6   # sensitivity factor
annual_cri['total_value'] = farm_area * value_per_ha

annual_cri['economic_loss_usd'] = (
    annual_cri['loss_percent']/100
) * annual_cri['total_value']

annual_cri.to_csv("step6_economic_loss.csv", index=False)

# =====================================================
# STEP 8: RISK CLASSIFICATION & INSURANCE
# =====================================================

print("\nSTEP 8: Insurance Risk Classification")

def risk_class(x):
    if x < 20:
        return "LOW"
    elif x < 40:
        return "MEDIUM"
    else:
        return "HIGH"

annual_cri['risk_level'] = annual_cri['loss_percent'].apply(risk_class)

base_rate = 0.05
multiplier = {'LOW':0.8,'MEDIUM':1.0,'HIGH':1.6}

annual_cri['insurance_premium'] = annual_cri.apply(
    lambda row: row['total_value'] * base_rate * multiplier[row['risk_level']],
    axis=1
)

annual_cri.to_csv("step7_insurance_report.csv", index=False)

# =====================================================
# STEP 9: MACHINE LEARNING (Risk Prediction)
# =====================================================

print("\nSTEP 9: Machine Learning Model")

features = weather_df.groupby('year').agg({
    'heavy_rain':'sum',
    'dry_spell':'sum',
    'heat_stress':'sum',
    'high_wind':'sum',
    'hail_event':'sum'
}).reset_index()

ml_df = features.merge(
    annual_cri[['year','risk_level']], on='year'
)

X = ml_df[['heavy_rain','dry_spell','heat_stress','high_wind','hail_event']]
y = ml_df['risk_level']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

pred = model.predict(X_test)
print("\nML Performance")
print(classification_report(y_test, pred))

importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': model.feature_importances_
})
importance.to_csv("step8_feature_importance.csv", index=False)

# =====================================================
# STEP 10: VISUALIZATION
# =====================================================

print("\nSTEP 10: Creating Graphs")

plt.figure(figsize=(10,5))
plt.plot(annual_cri['year'], annual_cri['CRI'], marker='o')
plt.title("Climate Risk Index Trend")
plt.xlabel("Year")
plt.ylabel("CRI")
plt.grid(True)
plt.savefig("graph_cri.png", dpi=300)
plt.show()

plt.figure(figsize=(10,5))
plt.plot(annual_cri['year'], annual_cri['economic_loss_usd'], marker='o')
plt.title("Economic Loss Trend")
plt.xlabel("Year")
plt.ylabel("Loss (USD)")
plt.grid(True)
plt.savefig("graph_loss.png", dpi=300)
plt.show()

plt.figure(figsize=(8,5))
plt.bar(importance['Feature'], importance['Importance'])
plt.title("Hazard Importance (ML)")
plt.savefig("graph_feature_importance.png", dpi=300)
plt.show()

print("\nModel Completed Successfully")