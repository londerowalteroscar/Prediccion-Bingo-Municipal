# 📦 Importaciones
import pandas as pd
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

# 📁 Ruta al archivo Excel
ruta_archivo = os.path.join("data", "tombola.xlsx")  # El script está en Scripts/

# 📥 Cargar datos
try:
    df = pd.read_excel(ruta_archivo)
    print("✅ Archivo cargado correctamente.")
except FileNotFoundError:
    print("❌ Error: archivo no encontrado.")
    exit()

# 🧹 Preprocesamiento
df.dropna(subset=["Fecha", "Numero"], inplace=True)
df["Fecha"] = pd.to_datetime(df["Fecha"], errors='coerce')
df.dropna(subset=["Fecha"], inplace=True)

# 🔄 Generar dataset binario
todos_los_numeros = list(range(0, 100))
fechas = sorted(df["Fecha"].dt.date.unique())

datos_binarios = []

for fecha in fechas:
    numeros_dia = df[df["Fecha"].dt.date == fecha]["Numero"].tolist()
    for numero in todos_los_numeros:
        datos_binarios.append({
            "fecha": fecha,
            "numero": numero,
            "salio": 1 if numero in numeros_dia else 0,
            "dia_semana": pd.Timestamp(fecha).weekday()  # 0: lunes, ..., 6: domingo
        })

df_binario = pd.DataFrame(datos_binarios)

# ➕ Feature: frecuencia histórica acumulada
frecuencia_historica = {num: 0 for num in todos_los_numeros}
frecuencias = []

for _, fila in df_binario.iterrows():
    num = fila["numero"]
    fecha = fila["fecha"]
    frecuencias.append(frecuencia_historica[num])
    if fila["salio"] == 1:
        frecuencia_historica[num] += 1

df_binario["frecuencia_pasada"] = frecuencias

# 🧪 Preparar datos para ML
X = df_binario[["numero", "dia_semana", "frecuencia_pasada"]]
y = df_binario["salio"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True, random_state=42)

# 🌳 Entrenar modelo Random Forest
modelo = RandomForestClassifier(n_estimators=100, random_state=42)
modelo.fit(X_train, y_train)

# 📊 Evaluación
y_pred = modelo.predict(X_test)
print("\n📋 Reporte de Clasificación:")
print(classification_report(y_test, y_pred))
print("🎯 Precisión general:", round(accuracy_score(y_test, y_pred) * 100, 2), "%")

# 🔍 Importancia de features
importancias = modelo.feature_importances_
nombres_features = X.columns.tolist()
print("\n📈 Importancia de las features:")
for nombre, importancia in zip(nombres_features, importancias):
    print(f"- {nombre}: {round(importancia * 100, 2)}%")

# 🔮 PREDICCIÓN: números con más probabilidad de salir en la próxima fecha
# Último día del dataset
ultima_fecha = max(df_binario["fecha"])
dia_semana = pd.Timestamp(ultima_fecha).weekday()

# Crear X_pred: una fila por cada número (0-99) con las últimas frecuencias
X_pred = pd.DataFrame({
    "numero": todos_los_numeros,
    "dia_semana": [dia_semana] * 100,
    "frecuencia_pasada": [frecuencia_historica[num] for num in todos_los_numeros]
})

# Predecir probabilidades
probas = modelo.predict_proba(X_pred)[:, 1]  # Probabilidad de que salga (salio == 1)
X_pred["probabilidad_salir"] = probas

# Obtener los 10 números con mayor probabilidad
top10 = X_pred.sort_values(by="probabilidad_salir", ascending=False).head(10)
mejores_numeros = top10["numero"].tolist()

print("\n🔟 Números con mayor probabilidad de salir en el próximo sorteo:")
print(mejores_numeros)

