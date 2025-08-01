# Prediccion-Bingo-Municipal

# 🎰 Predicción de Números Más Probables en Sorteos Semanales (Tómbola)

Este proyecto busca determinar los **10 números de dos cifras con mayor probabilidad empírica de salir** en sorteos diarios de lunes a sábado, utilizando datos históricos en el archivo `tombola.xlsx`.

## 📊 Naturaleza del Problema

* Tipo: **Análisis de eventos aleatorios**
* Objetivo: Identificar **tendencias empíricas** en la aparición de números.
* Dificultad: Los sorteos son eventos **teóricamente independientes y aleatorios**, lo que limita el uso de modelos de ML tradicionales.

---

## 🧠 Enfoques Posibles

### 🔹 Nivel 1 - Estadística Básica (Recomendado para empezar)

#### ✅ Análisis de Frecuencia

**Descripción**: Contar cuántas veces apareció cada número y seleccionar los 10 más frecuentes.

```python
import pandas as pd
df = pd.read_excel("tombola.xlsx")
top_10 = df['Numero'].value_counts().head(10)
print("Top 10 números más frecuentes:", top_10.index.tolist())
```

**Ventajas**:

* Simple, interpretable y rápido.
* Sin necesidad de muchos datos.

**Limitaciones**:

* No es predictivo en sentido estricto.
* Asume que patrones pasados pueden repetirse (puede no ser cierto en sorteos justos).

---

### 🔹 Nivel 2 - Modelos Intermedios con ML

#### 1. Clasificación Binaria (Random Forest / XGBoost / LightGBM)

**Descripción**: Crear un dataset donde por cada día indiques si un número apareció (1) o no (0), y uses features como día de la semana, frecuencia pasada, etc.

**Ventajas**:

* Puede modelar relaciones complejas entre features temporales y aparición de números.
* Funciona bien incluso con datos ruidosos.

**Limitaciones**:

* Puede sobreajustarse si los sorteos son verdaderamente aleatorios.

#### 2. Modelos de Series Temporales (Prophet / ARIMA)

**Descripción**: Tratar cada número (del 00 al 99) como una serie temporal binaria (1 si apareció, 0 si no) y modelar su aparición en el tiempo.

**Ventajas**:

* Ideal si hay tendencias o estacionalidades (por ejemplo, números que aparecen más ciertos días).

**Limitaciones**:

* Requiere muchos datos para ser efectivo.
* Dificultad para capturar la correlación entre números.

---

### 🔹 Nivel 3 - Modelos Avanzados

#### 1. LSTM / Transformers

**Descripción**: Modelar la secuencia completa de sorteos como una serie de secuencias (por día) y predecir la siguiente secuencia.

**Ventajas**:

* Captura relaciones complejas entre secuencias.

**Limitaciones**:

* Muy costoso computacionalmente.
* Necesita grandes volúmenes de datos.
* Difícil de justificar para eventos puramente aleatorios.

#### 2. Cadenas de Markov

**Descripción**: Modelar la probabilidad de aparición de un número en función de apariciones anteriores.

**Ventajas**:

* Buena si existe dependencia entre números.

**Limitaciones**:

* Los sorteos tienden a ser independientes, por lo tanto poco útil aquí.

#### 3. Regresión de Conteo (Poisson / Binomial Negativa)

**Descripción**: Modelar el conteo de aparición de cada número como una variable dependiente.

**Ventajas**:

* En teoría, útil para modelar datos de conteo.

**Limitaciones**:

* Difícil justificar causalidad en datos aleatorios.
* Supone distribución que puede no ajustarse a la realidad del sorteo.

#### 4. Clustering (K-Means / DBSCAN)

**Descripción**: Agrupar combinaciones de números que suelen salir juntas.

**Ventajas**:

* Puede revelar combinaciones recurrentes.

**Limitaciones**:

* No necesariamente mejora predicción futura.

---

## 🧾 Conclusiones y Recomendaciones

| Nivel | Técnica                            | ¿Recomendado?                  | Notas                             |
| ----- | ---------------------------------- | ------------------------------ | --------------------------------- |
| Bajo  | Frecuencia / Probabilidad empírica | ✅ Sí                           | Ideal como punto de partida       |
| Medio | Random Forest / XGBoost            | ✅ Sí                           | Requiere feature engineering      |
| Medio | Prophet / ARIMA                    | 🔄 Solo si hay tendencia clara |                                   |
| Alto  | LSTM / Transformers                | ⚠️ No necesario                | Sobreajuste muy probable          |
| Medio | Cadenas de Markov                  | ❌ No                           | Basado en supuesta dependencia    |
| Medio | Regresión de conteo                | ❌ Dudoso                       | Supone una causalidad inexistente |
| Medio | Clustering de combinaciones        | ❌ Exploratorio                 | No mejora la predicción real      |

---

## ⚠️ Advertencias

* ⚖️ **Eventos aleatorios**: Cada sorteo es teóricamente independiente.
* 🧠 **Falacia del jugador**: Que un número no haya salido no significa que "deba" salir.
* 🧪 **Sesgo físico**: Solo si hay defectos en el bolillero podría haber sesgos aprovechables.

---

## 🛠 Recomendación de Implementación Inicial

```python
import pandas as pd

df = pd.read_excel("tombola.xlsx")
top_numeros = df['Numero'].value_counts().head(10).index.tolist()
print("🎯 Top 10 números más frecuentes:", top_numeros)
```

Puedes hacer esto por semana, día de la semana o mes para capturar posibles tendencias temporales:

```python
df['Fecha'] = pd.to_datetime(df['Fecha'])
df['Día'] = df['Fecha'].dt.day_name()
top_por_dia = df.groupby(['Día', 'Numero']).size().groupby(level=0, group_keys=False).nlargest(10)
```

---

## 📂 Archivos del Proyecto

```
.
├── tombola.xlsx         # Datos históricos de sorteos
├── analisis_frecuencia.ipynb   # Análisis exploratorio + top 10
├── modelos_clasificacion.py    # (Opcional) Modelos de ML binario por número
└── README.md            # Descripción general del enfoque
```

---

## 🧠 ¿Y si quiero apostar?

Recuerda: este análisis **no garantiza predicciones certeras**. Es meramente exploratorio y debe tomarse como un estudio estadístico, **no como una guía infalible de apuestas**.

---
