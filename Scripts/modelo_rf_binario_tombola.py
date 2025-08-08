# 📦 Importaciones
import pandas as pd
import os
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from sklearn.metrics import classification_report, accuracy_score

# 📁 Ruta
ruta_excel = os.path.join("data", "tombola.xlsx")
ruta_salida = os.path.join("data", "modelo_rf_binario_tombola.csv")

# 📥 Cargar datos
try:
    df = pd.read_excel(ruta_excel)
    print("✅ Archivo cargado correctamente.")
except FileNotFoundError:
    print("❌ Error: archivo no encontrado.")
    exit()

# 🧹 Preprocesamiento
df.dropna(subset=["Fecha", "Numero"], inplace=True)
df["Fecha"] = pd.to_datetime(df["Fecha"], errors='coerce')
df.dropna(subset=["Fecha"], inplace=True)

# 🧱 Dataset binario
todos_los_numeros = list(range(100))
fechas = sorted(df["Fecha"].dt.date.unique())

datos_binarios = []
for fecha in fechas:
    numeros_dia = df[df["Fecha"].dt.date == fecha]["Numero"].tolist()
    for numero in todos_los_numeros:
        datos_binarios.append({
            "fecha": fecha,
            "numero": numero,
            "salio": 1 if numero in numeros_dia else 0,
            "dia_semana": pd.Timestamp(fecha).weekday()
        })

df_binario = pd.DataFrame(datos_binarios)

# 📅 Lógica de semanas
fecha_inicio = df["Fecha"].min().date()
fecha_fin = df["Fecha"].max().date()
inicio_primera_semana = fecha_inicio - timedelta(days=fecha_inicio.weekday())

# 📦 Resultados
resultados = []

fecha_actual = inicio_primera_semana
while fecha_actual <= fecha_fin:
    inicio_semana = fecha_actual
    fin_semana = inicio_semana + timedelta(days=6)
    siguiente_semana = fin_semana + timedelta(days=1)

    # Datos hasta fin de semana actual
    df_hist = df_binario[df_binario["fecha"] <= fin_semana]

    if df_hist.empty:
        resultados.append({
            "semana_inicio": inicio_semana.strftime("%Y-%m-%d"),
            "semana_fin": fin_semana.strftime("%Y-%m-%d"),
            "prediccion": ""
        })
        fecha_actual += timedelta(days=7)
        continue

    # Frecuencia acumulada
    frecuencia_historica = {num: 0 for num in todos_los_numeros}
    frecuencias = []
    for _, fila in df_hist.iterrows():
        num = fila["numero"]
        frecuencias.append(frecuencia_historica[num])
        if fila["salio"] == 1:
            frecuencia_historica[num] += 1
    df_hist["frecuencia_pasada"] = frecuencias

    # Entrenamiento
    X = df_hist[["numero", "dia_semana", "frecuencia_pasada"]]
    y = df_hist["salio"]
    if y.nunique() < 2:
        prediccion = ""
    else:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        modelo = RandomForestClassifier(n_estimators=100, random_state=42)
        modelo.fit(X_train, y_train)

        # Día de inicio de la próxima semana
        dia_pred = pd.Timestamp(siguiente_semana).weekday()
        X_pred = pd.DataFrame({
            "numero": todos_los_numeros,
            "dia_semana": [dia_pred] * 100,
            "frecuencia_pasada": [frecuencia_historica[num] for num in todos_los_numeros]
        })

        probas = modelo.predict_proba(X_pred)[:, 1]
        X_pred["probabilidad_salir"] = probas

        top10 = X_pred.sort_values(by="probabilidad_salir", ascending=False).head(10)
        prediccion = str(top10["numero"].tolist())

    resultados.append({
        "semana_inicio": inicio_semana.strftime("%Y-%m-%d"),
        "semana_fin": fin_semana.strftime("%Y-%m-%d"),
        "prediccion": prediccion
    })

    fecha_actual += timedelta(days=7)

# 💾 Guardar CSV
df_resultado = pd.DataFrame(resultados)
df_resultado.to_csv(ruta_salida, index=False)
print(f"\n✅ Resultados guardados en: {ruta_salida}")
