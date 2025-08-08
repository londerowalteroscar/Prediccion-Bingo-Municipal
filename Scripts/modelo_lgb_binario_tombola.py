# 📦 Importaciones
import pandas as pd
import os
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from datetime import datetime, timedelta

# 📁 Rutas
ruta_excel = os.path.join("data", "tombola.xlsx")
ruta_csv = os.path.join("data", "modelo_lgb_binario_tombola.csv")

# 📥 Cargar datos
try:
    df = pd.read_excel(ruta_excel)
    print("✅ Archivo cargado correctamente.")
except FileNotFoundError:
    print("❌ Error: archivo no encontrado.")
    exit()

# 🧹 Preprocesamiento inicial
df.dropna(subset=["Fecha", "Numero"], inplace=True)
df["Fecha"] = pd.to_datetime(df["Fecha"], errors='coerce')
df.dropna(subset=["Fecha"], inplace=True)
df["Numero"] = df["Numero"].astype(int)

# 🎯 Variables auxiliares
todos_los_numeros = list(range(0, 100))
fecha_inicio = df["Fecha"].min().date()
fecha_fin = df["Fecha"].max().date()
inicio_primera_semana = fecha_inicio - timedelta(days=fecha_inicio.weekday())

# 📦 Lista para almacenar resultados semanales
resultados = []

fecha_actual = inicio_primera_semana
while fecha_actual <= fecha_fin:
    inicio_semana = fecha_actual
    fin_semana = inicio_semana + timedelta(days=6)
    fin_semana_datetime = datetime.combine(fin_semana, datetime.min.time())

    # 📊 Filtrar datos hasta el final de esa semana
    df_semanal = df[df["Fecha"] <= fin_semana_datetime]
    if df_semanal.empty:
        resultados.append({
            "semana_inicio": inicio_semana.strftime('%Y-%m-%d'),
            "semana_fin": fin_semana.strftime('%Y-%m-%d'),
            "prediccion": ""
        })
        fecha_actual += timedelta(days=7)
        continue

    fechas = sorted(df_semanal["Fecha"].dt.date.unique())
    datos_binarios = []
    frecuencia_historica = {num: 0 for num in todos_los_numeros}

    # ⚙️ Generar dataset binario histórico
    for fecha in fechas:
        numeros_dia = df_semanal[df_semanal["Fecha"].dt.date == fecha]["Numero"].tolist()
        for numero in todos_los_numeros:
            datos_binarios.append({
                "fecha": fecha,
                "numero": numero,
                "salio": 1 if numero in numeros_dia else 0,
                "dia_semana": pd.Timestamp(fecha).weekday(),
                "frecuencia_pasada": frecuencia_historica[numero]
            })
        for num in numeros_dia:
            frecuencia_historica[num] += 1

    df_binario = pd.DataFrame(datos_binarios)

    # Entrenar modelo
    X = df_binario[["numero", "dia_semana", "frecuencia_pasada"]]
    y = df_binario["salio"]

    if y.nunique() < 2:
        print(f"⚠️ Semana {inicio_semana} - No hay variedad de clases para entrenar.")
        resultados.append({
            "semana_inicio": inicio_semana.strftime('%Y-%m-%d'),
            "semana_fin": fin_semana.strftime('%Y-%m-%d'),
            "prediccion": ""
        })
        fecha_actual += timedelta(days=7)
        continue

    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, shuffle=True, random_state=42)
    modelo = lgb.LGBMClassifier(n_estimators=100, learning_rate=0.1, random_state=42)
    modelo.fit(X_train, y_train)

    # 🔮 Predecir para la semana siguiente
    siguiente_semana = fin_semana + timedelta(days=1)
    dia_semana = siguiente_semana.weekday()

    X_pred = pd.DataFrame({
        "numero": todos_los_numeros,
        "dia_semana": [dia_semana] * 100,
        "frecuencia_pasada": [frecuencia_historica[num] for num in todos_los_numeros]
    })

    probas = modelo.predict_proba(X_pred)[:, 1]
    X_pred["probabilidad_salir"] = probas

    top10 = X_pred.sort_values(by="probabilidad_salir", ascending=False).head(10)
    mejores_numeros = top10["numero"].tolist()

    # ✅ Guardar predicción semanal
    resultados.append({
        "semana_inicio": inicio_semana.strftime('%Y-%m-%d'),
        "semana_fin": fin_semana.strftime('%Y-%m-%d'),
        "prediccion": str(mejores_numeros)
    })

    # ➡️ Avanzar a la siguiente semana
    fecha_actual += timedelta(days=7)

# 💾 Guardar resultados
df_resultado = pd.DataFrame(resultados)
df_resultado.to_csv(ruta_csv, index=False)
print(f"\n✅ Predicciones semanales guardadas en '{ruta_csv}'")
