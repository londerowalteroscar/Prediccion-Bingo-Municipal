# 📦 Importaciones
import pandas as pd
from prophet import Prophet
import os
import logging
import warnings
from datetime import datetime, timedelta

# 🔕 Silenciar logs
logging.getLogger("cmdstanpy").setLevel(logging.CRITICAL)
logging.getLogger("prophet").setLevel(logging.CRITICAL)
warnings.filterwarnings("ignore")

# 📁 Rutas
ruta_archivo = os.path.join("data", "tombola.xlsx")
ruta_csv = os.path.join("data", "modelo_prophet_tombola.csv")

# 📥 Cargar datos
try:
    df = pd.read_excel(ruta_archivo)
    print("✅ Archivo cargado correctamente.")
except FileNotFoundError:
    print("❌ Error: archivo no encontrado.")
    exit()

# 🧹 Preprocesamiento
df.dropna(subset=["Numero", "Fecha"], inplace=True)
df["Fecha"] = pd.to_datetime(df["Fecha"])
df["Numero"] = df["Numero"].astype(int)

# 📅 Definir el rango de semanas
fecha_inicio = df["Fecha"].min().date()
fecha_fin = df["Fecha"].max().date()
inicio_primera_semana = fecha_inicio - timedelta(days=fecha_inicio.weekday())

# 📦 Resultados semanales
resultados = []

fecha_actual = inicio_primera_semana
while fecha_actual <= fecha_fin:
    inicio_semana = fecha_actual
    fin_semana = inicio_semana + timedelta(days=6)
    fin_semana_dt = datetime.combine(fin_semana, datetime.min.time())
    
    # 🔍 Filtrar datos hasta el fin de la semana
    df_hist = df[df["Fecha"] <= fin_semana_dt]
    
    if df_hist.empty:
        resultados.append({
            "semana_inicio": inicio_semana.strftime('%Y-%m-%d'),
            "semana_fin": fin_semana.strftime('%Y-%m-%d'),
            "prediccion": ""
        })
        fecha_actual += timedelta(days=7)
        continue

    # 🔮 Modelar con Prophet para cada número
    promedios_yhat = {}
    
    for numero in range(100):
        df_num = df_hist.copy()
        df_num["y"] = (df_num["Numero"] == numero).astype(int)
        df_num = df_num.groupby("Fecha").agg({"y": "sum"}).reset_index()
        df_num.rename(columns={"Fecha": "ds"}, inplace=True)

        if df_num["y"].sum() < 3:
            continue  # muy pocos datos

        try:
            modelo = Prophet(daily_seasonality=True, weekly_seasonality=True, yearly_seasonality=False)
            modelo.fit(df_num)

            futuro = modelo.make_future_dataframe(periods=6)
            pred = modelo.predict(futuro)

            yhat_promedio = pred.tail(6)["yhat"].mean()
            promedios_yhat[numero] = yhat_promedio
        except:
            continue

    # 📊 Top 10 de esa semana
    if promedios_yhat:
        top10 = sorted(promedios_yhat, key=promedios_yhat.get, reverse=True)[:10]
        prediccion = str(top10)
    else:
        prediccion = ""

    # 💾 Guardar resultados
    resultados.append({
        "semana_inicio": inicio_semana.strftime('%Y-%m-%d'),
        "semana_fin": fin_semana.strftime('%Y-%m-%d'),
        "prediccion": prediccion
    })

    # ➡️ Avanzar una semana
    fecha_actual += timedelta(days=7)

# 📤 Guardar archivo CSV
df_resultado = pd.DataFrame(resultados)
df_resultado.to_csv(ruta_csv, index=False)
print(f"\n✅ Resultados semanales guardados en '{ruta_csv}'")
