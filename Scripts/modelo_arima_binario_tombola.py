import os
import pandas as pd
import numpy as np
import warnings
from statsmodels.tsa.arima.model import ARIMA

warnings.filterwarnings("ignore")

# Ruta
ruta_archivo = os.path.join("data", "tombola.xlsx")

# Cargar archivo
df = pd.read_excel(ruta_archivo)

# Usar la columna correcta 'Fecha' con mayúscula y convertirla a datetime
df['Fecha'] = pd.to_datetime(df['Fecha'])
df = df.sort_values('Fecha')

# Crear DataFrame base con fechas
fechas = pd.date_range(start=df['Fecha'].min(), end=df['Fecha'].max(), freq='D')
df_base = pd.DataFrame({'Fecha': fechas})

# Lista de números a analizar
numeros = range(100)
predicciones = []

for numero in numeros:
    df_num = df.copy()
    df_num['presente'] = df_num['Numero'].apply(lambda x: 1 if x == numero else 0)
    
    # Agrupar por fecha si un número apareció ese día
    df_diario = df_num.groupby('Fecha')['presente'].max().reset_index()
    df_serie = pd.merge(df_base, df_diario, on='Fecha', how='left').fillna(0)
    df_serie.set_index('Fecha', inplace=True)

    # Solo números con al menos 10 apariciones
    if df_serie['presente'].sum() >= 10:
        try:
            modelo = ARIMA(df_serie['presente'], order=(2,0,2)).fit()
            pred = modelo.forecast(steps=5).mean()
            probabilidad = pred
            predicciones.append((numero, probabilidad))
        except:
            continue

# Ordenar y obtener top 10
predicciones.sort(key=lambda x: x[1], reverse=True)
numeros_probables = [num for num, _ in predicciones[:10]]

print("📊 Predicción de aparición para los próximos días con ARIMA:")
print(numeros_probables)
