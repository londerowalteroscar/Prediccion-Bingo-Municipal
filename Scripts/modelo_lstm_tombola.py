import os
import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Embedding, Input
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from datetime import datetime, timedelta

# 📁 Cargar archivo y preparar datos para el modelo LSTM
ruta_archivo = os.path.join("data", "tombola.xlsx")
try:
    df = pd.read_excel(ruta_archivo)
except FileNotFoundError:
    print("❌ Error: No se encontró el archivo 'tombola.xlsx' en la carpeta 'data'.")
    exit()

df['Fecha'] = pd.to_datetime(df['Fecha'])
df.dropna(subset=["Numero"], inplace=True)  # Eliminar filas sin número

secuencias = df.groupby('Fecha')['Numero'].apply(list).reset_index()
secuencias = secuencias[secuencias['Numero'].apply(len) == 10]

X = []
y = []
for secuencia in secuencias['Numero']:
    for i in range(len(secuencia) - 1):
        X.append(secuencia[i])
        y.append(secuencia[i + 1])

todos_los_numeros = sorted(list(set(X + y)))
encoder = LabelEncoder()
encoder.fit(todos_los_numeros)

X_encoded = encoder.transform(X)
y_encoded = encoder.transform(y)
y_encoded = to_categorical(y_encoded, num_classes=len(encoder.classes_))

# 🧠 Definir y entrenar modelo LSTM
model = Sequential()
model.add(Input(shape=(1,)))
model.add(Embedding(input_dim=len(encoder.classes_), output_dim=64))
model.add(LSTM(64))
model.add(Dense(len(encoder.classes_), activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

X_encoded = np.array(X_encoded)
model.fit(X_encoded, y_encoded, epochs=20, verbose=0)


# 💾 Guardar el modelo entrenado (opcional pero recomendado)
ruta_modelo = os.path.join("data", "modelo_lstm_tombola.h5")  # Ajusta la ruta si es necesario
model.save(ruta_modelo)

# --- Predicciones Semanales y CSV ---

ruta_csv = os.path.join("data", "modelo_lstm_tombola.csv") # Nombre del archivo CSV

fecha_inicio_datos = df["Fecha"].min().date()
fecha_fin_datos = df["Fecha"].max().date()
inicio_primera_semana = fecha_inicio_datos - timedelta(days=fecha_inicio_datos.weekday())

lista_resultados = []
fecha_actual_prediccion = inicio_primera_semana

while fecha_actual_prediccion <= fecha_fin_datos:
    inicio_semana = fecha_actual_prediccion
    fin_semana = inicio_semana + timedelta(days=6)
    fin_semana_datetime = datetime.combine(fin_semana, datetime.min.time())

    df_filtrado = df[df["Fecha"] <= fin_semana_datetime]

    # Obtener el último número de la secuencia para hacer la predicción
    ultimo_numero = df_filtrado['Numero'].iloc[-1][-1]  # Último número de la última secuencia

    entrada = np.array([encoder.transform([ultimo_numero])[0]])

    predicciones = []
    for _ in range(10):
        pred = model.predict(entrada, verbose=0)
        pred_num = encoder.inverse_transform([np.argmax(pred)])
        predicciones.append(pred_num[0])
        entrada = np.array([encoder.transform([pred_num[0]])[0]])

    # Convertir las predicciones a strings para guardarlas en el CSV
    prediccion_str = str(predicciones)

    nuevo_registro = {
        'semana_inicio': inicio_semana.strftime('%Y-%m-%d'),
        'semana_fin': fin_semana.strftime('%Y-%m-%d'),
        'prediccion': prediccion_str
    }
    lista_resultados.append(nuevo_registro)

    fecha_actual_prediccion += timedelta(days=7)

df_predicciones = pd.DataFrame(lista_resultados)

if os.path.exists(ruta_csv):
    print("⚠️ El archivo de predicciones ya existía y será sobrescrito.")

df_predicciones.to_csv(ruta_csv, index=False)

print(f"✅ Se han generado predicciones semanales con LSTM y se han guardado en '{ruta_csv}'")
print(f"📅 Total de predicciones generadas: {len(df_predicciones)}")
