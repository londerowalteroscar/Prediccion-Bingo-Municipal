import os
import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Embedding, Input
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical

# 📁 Cargar archivo
ruta_archivo = os.path.join("data", "tombola.xlsx")
df = pd.read_excel(ruta_archivo)
df['Fecha'] = pd.to_datetime(df['Fecha'])

# 🔢 Agrupar secuencias por fecha
secuencias = df.groupby('Fecha')['Numero'].apply(list).reset_index()
secuencias = secuencias[secuencias['Numero'].apply(len) == 10]

# 📊 Preparar datos supervisados
X = []
y = []

for secuencia in secuencias['Numero']:
    for i in range(len(secuencia) - 1):
        X.append(secuencia[i])
        y.append(secuencia[i + 1])

# 🔠 Codificación LabelEncoder
todos_los_numeros = sorted(list(set(X + y)))
encoder = LabelEncoder()
encoder.fit(todos_los_numeros)

X_encoded = encoder.transform(X)
y_encoded = encoder.transform(y)
y_encoded = to_categorical(y_encoded, num_classes=len(encoder.classes_))

# 🧠 Definir modelo LSTM
model = Sequential()
model.add(Input(shape=(1,)))
model.add(Embedding(input_dim=len(encoder.classes_), output_dim=64))
model.add(LSTM(64))
model.add(Dense(len(encoder.classes_), activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# 🏋️‍♂️ Entrenar
X_encoded = np.array(X_encoded)
model.fit(X_encoded, y_encoded, epochs=20, verbose=0)

# 🔮 Predicción de los próximos 10 números
ultimo_numero = secuencias['Numero'].iloc[-1][-1]
entrada = np.array([encoder.transform([ultimo_numero])[0]])

predicciones = []
for _ in range(10):
    pred = model.predict(entrada, verbose=0)
    pred_num = encoder.inverse_transform([np.argmax(pred)])
    predicciones.append(pred_num[0])
    entrada = np.array([encoder.transform([pred_num[0]])[0]])

print("\n🔮 Predicción de los próximos 10 números con LSTM:")
print([int(num) for num in predicciones])
