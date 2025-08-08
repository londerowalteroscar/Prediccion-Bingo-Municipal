# 📦 Importación de librerías
import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import os

# 📁 Ruta al archivo
ruta_archivo = os.path.join("data", "tombola.xlsx")

# 📥 Cargar archivo
df = pd.read_excel(ruta_archivo)
df.dropna(subset=["Numero", "Fecha"], inplace=True)

# 🧹 Preprocesamiento
df["Fecha"] = pd.to_datetime(df["Fecha"])
df["dia_semana"] = df["Fecha"].dt.dayofweek
df["Numero"] = df["Numero"].astype(int)

# 📊 Crear tabla dinámica: número vs. frecuencia por día de la semana
tabla = pd.crosstab(df["Numero"], df["dia_semana"])

# ⚖️ Escalar los datos
scaler = StandardScaler()
X_scaled = scaler.fit_transform(tabla)

# 📌 Aplicar DBSCAN
dbscan = DBSCAN(eps=1.5, min_samples=2)
labels = dbscan.fit_predict(X_scaled)

# ➕ Añadir etiquetas a los datos
tabla["cluster"] = labels

# 📤 Filtrar el cluster principal (el más numeroso que no sea ruido -1)
cluster_counts = tabla["cluster"].value_counts()
cluster_principal = cluster_counts[cluster_counts.index != -1].idxmax()

# 🏆 Top 10 números más frecuentes dentro del cluster principal
numeros_cluster = tabla[tabla["cluster"] == cluster_principal].drop("cluster", axis=1)
top_10_dbscan = numeros_cluster.sum(axis=1).sort_values(ascending=False).head(10).index.tolist()

# ✅ Resultado
print("🔍 Predicción por agrupamiento DBSCAN (detectando densidad):")
print(top_10_dbscan)
