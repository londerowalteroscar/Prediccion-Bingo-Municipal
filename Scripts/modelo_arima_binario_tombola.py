# 📦 Importaciones
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
from statsmodels.tsa.arima.model import ARIMA

# 🚫 Silenciar warnings
warnings.filterwarnings("ignore")

# 📁 Rutas
ruta_excel = os.path.join("data", "tombola.xlsx")
ruta_csv = os.path.join("data", "modelo_arima_binario_tombola.csv")

# 📥 Cargar datos
try:
    df = pd.read_excel(ruta_excel)
    print("✅ Archivo cargado correctamente.")
except FileNotFoundError:
    print("❌ Error: No se encontró el archivo 'tombola.xlsx'.")
    exit()

# 🧹 Preprocesamiento
df.dropna(subset=["Fecha", "Numero"], inplace=True)
df["Fecha"] = pd.to_datetime(df["Fecha"])
df.sort_values("Fecha", inplace=True)

# 📅 Rango de semanas
fecha_inicio = df["Fecha"].min().date()
fecha_fin = df["Fecha"].max().date()
inicio_primera_semana = fecha_inicio - timedelta(days=fecha_inicio.weekday())

# 📦 Lista de resultados
resultados = []

fecha_actual = inicio_primera_semana
while fecha_actual <= fecha_fin:
    inicio_semana = fecha_actual
    fin_semana = inicio_semana + timedelta(days=6)
    fin_semana_dt = datetime.combine(fin_semana, datetime.min.time())

    # Filtrar datos hasta el fin de esa semana
    df_hist = df[df["Fecha"] <= fin_semana_dt]

    if df_hist.empty:
        resultados.append({
            "semana_inicio": inicio_semana.strftime("%Y-%m-%d"),
            "semana_fin": fin_semana.strftime("%Y-%m-%d"),
            "prediccion": ""
        })
        fecha_actual += timedelta(days=7)
        continue

    # Crear base de fechas para ARIMA
    fechas = pd.date_range(start=df_hist["Fecha"].min(), end=df_hist["Fecha"].max(), freq='D')
    df_base = pd.DataFrame({'Fecha': fechas})

    predicciones = []

    for numero in range(100):
        df_num = df_hist.copy()
        df_num["presente"] = df_num["Numero"].apply(lambda x: 1 if x == numero else 0)

        df_diario = df_num.groupby("Fecha")["presente"].max().reset_index()
        df_serie = pd.merge(df_base, df_diario, on="Fecha", how="left").fillna(0)
        df_serie.set_index("Fecha", inplace=True)

        # Requiere al menos 10 días donde haya salido
        if df_serie["presente"].sum() >= 3:
            try:
                modelo = ARIMA(df_serie["presente"], order=(2,0,2)).fit()
                pred = modelo.forecast(steps=6).mean()
                predicciones.append((numero, pred))
            except:
                continue

    # Ordenar y seleccionar top 10
    if predicciones:
        predicciones.sort(key=lambda x: x[1], reverse=True)
        top_10 = [num for num, _ in predicciones[:10]]
        prediccion_str = str(top_10)
    else:
        prediccion_str = ""

    # Agregar resultado semanal
    resultados.append({
        "semana_inicio": inicio_semana.strftime("%Y-%m-%d"),
        "semana_fin": fin_semana.strftime("%Y-%m-%d"),
        "prediccion": prediccion_str
    })

    # Avanzar a la siguiente semana
    fecha_actual += timedelta(days=7)

# 💾 Guardar CSV
df_resultado = pd.DataFrame(resultados)
df_resultado.to_csv(ruta_csv, index=False)
print(f"\n✅ Resultados guardados en: {ruta_csv}")
