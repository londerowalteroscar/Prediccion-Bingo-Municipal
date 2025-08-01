# 📦 Importación de librerías
import pandas as pd
from prophet import Prophet
from datetime import timedelta
import os
import logging
import sys

# 🧹 Silenciar logs innecesarios de Prophet y cmdstanpy
logging.getLogger("cmdstanpy").setLevel(logging.CRITICAL)
logging.getLogger("prophet").setLevel(logging.CRITICAL)
logging.getLogger("matplotlib").setLevel(logging.CRITICAL)

# Desactivar warnings de consola
import warnings
warnings.filterwarnings("ignore")

# 📁 Ruta del archivo Excel
ruta_archivo = os.path.join("data", "tombola.xlsx")

# 📥 Cargar el archivo
df = pd.read_excel(ruta_archivo)

# Preprocesamiento
df.dropna(subset=["Numero", "Fecha"], inplace=True)
df["Fecha"] = pd.to_datetime(df["Fecha"])
df["Numero"] = df["Numero"].astype(int)

# 🔮 Predicción con Prophet
numeros_probables = []

for numero in range(100):
    df_num = df.copy()
    df_num["y"] = (df_num["Numero"] == numero).astype(int)
    df_num = df_num.groupby("Fecha").agg({"y": "sum"}).reset_index()
    df_num.rename(columns={"Fecha": "ds"}, inplace=True)

    if df_num["y"].sum() < 3:
        continue

    try:
        modelo = Prophet(daily_seasonality=True, yearly_seasonality=False, weekly_seasonality=True)
        modelo.fit(df_num)

        futuro = modelo.make_future_dataframe(periods=6)
        pred = modelo.predict(futuro)

        if pred.tail(6)["yhat"].sum() > 0.5:
            numeros_probables.append(numero)
    except:
        continue

# ✅ Salida limpia
print("📅 Predicción de aparición para los próximos días con Prophet:")
print(numeros_probables)
