# 📦 Importación de librerías
import pandas as pd
import os
from collections import defaultdict, Counter
from datetime import timedelta

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
for _, grupo in df.groupby("Fecha"):
    numeros_dia = grupo["Numero"].tolist()
    for i in range(len(numeros_dia) - 1):
        transiciones[numeros_dia[i]].append(numeros_dia[i + 1])

# 🔢 Contar transiciones
matriz_markov = {num: Counter(siguientes) for num, siguientes in transiciones.items()}

# 📅 Generar predicciones semana por semana
predicciones_semana = []
fechas_inicio = pd.date_range(df["Fecha"].min(), df["Fecha"].max(), freq="W-MON")

for inicio in fechas_inicio:
    fin = inicio + timedelta(days=6)

    # Tomar el último número antes o en la semana actual
    df_semana = df[df["Fecha"] <= fin]
    if df_semana.empty:
        continue
    ultimo_numero = df_semana.iloc[-1]["Numero"]

    # Obtener predicciones iniciales (máximo 10) de la matriz Markov
    if ultimo_numero in matriz_markov:
        prediccion = [num for num, _ in matriz_markov[ultimo_numero].most_common(10)]
    else:
        prediccion = []

    # Completar hasta 10 números usando los más frecuentes globales
    top_global = df["Numero"].value_counts().index.tolist()
    for num in top_global:
        if num not in prediccion:
            prediccion.append(num)
        if len(prediccion) == 10:
            break

    predicciones_semana.append({
        "semana_inicio": inicio.date(),
        "semana_fin": fin.date(),
        "prediccion": str(prediccion)  # Guardar como lista con corchetes
    })

# 💾 Guardar en CSV
csv_path = os.path.join("data", "modelo_markov_tombola.csv")
pd.DataFrame(predicciones_semana).to_csv(csv_path, index=False)

# 📤 Mostrar última predicción
print("📅 Última predicción generada:")
print(predicciones_semana[-1])
