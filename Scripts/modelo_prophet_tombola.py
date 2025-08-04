# 📦 Importación de librerías
import pandas as pd
from prophet import Prophet
import os
import logging
import warnings

# 🔕 Silenciar logs innecesarios
logging.getLogger("cmdstanpy").setLevel(logging.CRITICAL)
logging.getLogger("prophet").setLevel(logging.CRITICAL)
warnings.filterwarnings("ignore")

# 📁 Ruta al archivo
ruta_archivo = os.path.join("data", "tombola.xlsx")

# 📥 Cargar datos
df = pd.read_excel(ruta_archivo)
df.dropna(subset=["Numero", "Fecha"], inplace=True)
df["Fecha"] = pd.to_datetime(df["Fecha"])
df["Numero"] = df["Numero"].astype(int)

# 🔮 Inicialización de modelo Prophet por número
modelo_por_numero = {}
promedios_yhat = {}

for numero in range(100):
    df_num = df.copy()
    df_num["y"] = (df_num["Numero"] == numero).astype(int)
    df_num = df_num.groupby("Fecha").agg({"y": "sum"}).reset_index()
    df_num.rename(columns={"Fecha": "ds"}, inplace=True)

    if df_num["y"].sum() < 3:
        continue  # evitar ruido con pocos datos

    try:
        modelo = Prophet(daily_seasonality=True, weekly_seasonality=True, yearly_seasonality=False)
        modelo.fit(df_num)

        futuro = modelo.make_future_dataframe(periods=6)
        pred = modelo.predict(futuro)

        yhat_promedio = pred.tail(6)["yhat"].mean()
        promedios_yhat[numero] = yhat_promedio
    except:
        continue

# 📊 Obtener top 10 números con mayor promedio de predicción
top_10_numeros = sorted(promedios_yhat, key=promedios_yhat.get, reverse=True)[:10]

# ✅ Mostrar resultado final
print("📅 Predicción de aparición para los próximos días con Prophet:")
print(top_10_numeros)
