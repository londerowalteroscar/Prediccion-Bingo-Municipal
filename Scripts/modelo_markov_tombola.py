# 📦 Importación de librerías
import pandas as pd
import os
from collections import defaultdict, Counter
import random

# 📁 Ruta del archivo
ruta_archivo = os.path.join("data", "tombola.xlsx")

# 📥 Cargar el archivo
df = pd.read_excel(ruta_archivo)
df.dropna(subset=["Numero", "Fecha"], inplace=True)
df["Fecha"] = pd.to_datetime(df["Fecha"])
df["Numero"] = df["Numero"].astype(int)

# 🔁 Ordenar por fecha para mantener el orden cronológico
df = df.sort_values("Fecha")

# 🔧 Crear matriz de transición de Markov
transiciones = defaultdict(list)

# Agrupar por fecha, y registrar secuencias de números por día
for _, grupo in df.groupby("Fecha"):
    numeros_dia = grupo["Numero"].tolist()
    for i in range(len(numeros_dia) - 1):
        transiciones[numeros_dia[i]].append(numeros_dia[i + 1])

# 🔢 Contar transiciones para cada número
matriz_markov = {num: Counter(siguientes) for num, siguientes in transiciones.items()}

# 🔮 Generar predicciones: partir del último número sorteado
ultimo_numero = df.iloc[-1]["Numero"]
predicciones = []

# Obtener las transiciones más probables desde el último número
if ultimo_numero in matriz_markov and len(matriz_markov[ultimo_numero]) >= 3:
    probables = matriz_markov[ultimo_numero].most_common(10)
    predicciones = [num for num, _ in probables]
else:
    # Mezcla entre transiciones (si hay) y los más frecuentes globalmente
    prob_transiciones = matriz_markov.get(ultimo_numero, {}).most_common()
    predicciones = [num for num, _ in prob_transiciones]

    # Rellenar con los más frecuentes si faltan
    faltantes = 10 - len(predicciones)
    if faltantes > 0:
        top_global = df["Numero"].value_counts().index.tolist()
        for num in top_global:
            if num not in predicciones:
                predicciones.append(num)
            if len(predicciones) == 10:
                break

# ✅ Mostrar predicciones
print("🔗 Predicción con Cadenas de Markov (desde el último número sorteado):")
print(predicciones[:10])