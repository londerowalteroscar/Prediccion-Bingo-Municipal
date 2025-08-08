# 📦 Importación de librerías
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
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

# 🔢 Aplicar K-Means (con 3 clusters, puedes ajustar)
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
clusters = kmeans.fit_predict(tabla)

# ➕ Agregar los clusters a la tabla
tabla["cluster"] = clusters

# 📌 Elegimos el cluster con la mayor frecuencia promedio
cluster_promedios = tabla.groupby("cluster").sum().mean(axis=1)
mejor_cluster = cluster_promedios.idxmax()

# 🏆 Seleccionar los 10 números con mayor frecuencia dentro del mejor cluster
numeros_cluster = tabla[tabla["cluster"] == mejor_cluster].drop("cluster", axis=1)
top_10_kmeans = numeros_cluster.sum(axis=1).sort_values(ascending=False).head(10).index.tolist()

# ✅ Resultado
print("🧠 Predicción por agrupamiento K-Means (basado en frecuencia por día):")
print(top_10_kmeans)
