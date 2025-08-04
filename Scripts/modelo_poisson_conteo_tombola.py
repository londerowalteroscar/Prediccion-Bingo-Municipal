# 📦 Importación de librerías
import pandas as pd
import numpy as np
import statsmodels.api as sm
import os

# 📁 Ruta del archivo
ruta_archivo = os.path.join("data", "tombola.xlsx")

# 📥 Carga del archivo
df = pd.read_excel(ruta_archivo)
df.dropna(subset=["Numero", "Fecha"], inplace=True)

# 🧹 Preprocesamiento
df["Fecha"] = pd.to_datetime(df["Fecha"])
df["dia_semana"] = df["Fecha"].dt.dayofweek
df["Numero"] = df["Numero"].astype(int)

# 📊 Crear dataset agregado con conteos por número y día de la semana
conteo_df = df.groupby(["Numero", "dia_semana"]).size().reset_index(name="conteo")

# 🔄 Expandir todos los números del 0 al 99 y días de la semana 0–6 (para evitar valores faltantes)
todos_numeros = pd.DataFrame([(n, d) for n in range(100) for d in range(7)], columns=["Numero", "dia_semana"])
conteo_df = todos_numeros.merge(conteo_df, on=["Numero", "dia_semana"], how="left").fillna(0)
conteo_df["conteo"] = conteo_df["conteo"].astype(int)

# 🧠 Variables independientes (con constante)
X = conteo_df[["Numero", "dia_semana"]]
X = sm.add_constant(X)

# 🎯 Variable dependiente
y = conteo_df["conteo"]

# ⚙️ Modelo de regresión de Poisson
modelo = sm.GLM(y, X, family=sm.families.Poisson())
resultado = modelo.fit()

# 🔍 Predecir conteos futuros (esperados)
conteo_df["predicho"] = resultado.predict(X)

# 📈 Promedio esperado de aparición por número (suma sobre días)
predicciones_agrupadas = conteo_df.groupby("Numero")["predicho"].sum().sort_values(ascending=False)

# 🏆 Top 10 números más probables de aparecer
top_10_poisson = predicciones_agrupadas.head(10).index.tolist()

# ✅ Resultado
print("📈 Predicción de aparición por Regresión de Poisson:")
print(top_10_poisson)
