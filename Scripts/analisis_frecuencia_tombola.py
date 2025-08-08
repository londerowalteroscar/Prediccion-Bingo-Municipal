import pandas as pd
import os
from datetime import datetime, timedelta

# Ruta al archivo Excel
ruta_excel = os.path.join("data", "tombola.xlsx")

# Ruta donde se guardará el archivo CSV de predicciones
ruta_csv = os.path.join("data", "analisis_frecuencia_tombola.csv")

# Cargar el archivo Excel
try:
    df = pd.read_excel(ruta_excel)
except FileNotFoundError:
    print("❌ Error: No se encontró el archivo 'tombola.xlsx' en la carpeta 'data'.")
    exit()

# Eliminar filas sin número y asegurar que la columna 'Fecha' sea de tipo datetime
df.dropna(subset=["Numero"], inplace=True)
df["Fecha"] = pd.to_datetime(df["Fecha"])

# Obtener el rango de fechas del DataFrame
fecha_inicio_datos = df["Fecha"].min().date()
fecha_fin_datos = df["Fecha"].max().date()

# Calcular el inicio de la primera semana completa en los datos
inicio_primera_semana = fecha_inicio_datos - timedelta(days=fecha_inicio_datos.weekday())

# Lista para almacenar los resultados
lista_resultados = []

# Iterar semana a semana desde el inicio de los datos hasta el final
fecha_actual_prediccion = inicio_primera_semana
while fecha_actual_prediccion <= fecha_fin_datos:
    # Definir el rango de la semana para la predicción
    inicio_semana = fecha_actual_prediccion
    fin_semana = inicio_semana + timedelta(days=6)

    # Convertir fin_semana a datetime para que coincida con el tipo de la columna 'Fecha'
    fin_semana_datetime = datetime.combine(fin_semana, datetime.min.time())

    # Filtrar los datos que están disponibles hasta el final de la semana actual
    df_filtrado = df[df["Fecha"] <= fin_semana_datetime]

    # Calcular frecuencias con los datos históricos hasta esa semana
    frecuencia = df_filtrado["Numero"].value_counts().sort_values(ascending=False)

    # Obtener el top 10 (la predicción para la siguiente semana)
    prediccion_semanal = frecuencia.head(10).index.tolist()

    # Preparar el nuevo registro y añadirlo a la lista
    nuevo_registro = {
        'semana_inicio': inicio_semana.strftime('%Y-%m-%d'),
        'semana_fin': fin_semana.strftime('%Y-%m-%d'),
        'prediccion': str(prediccion_semanal)
    }
    lista_resultados.append(nuevo_registro)

    # Pasar a la siguiente semana
    fecha_actual_prediccion += timedelta(days=7)

# Crear el DataFrame final a partir de la lista de resultados
df_predicciones = pd.DataFrame(lista_resultados)

# Verificar si el archivo ya existía (solo para informar)
if os.path.exists(ruta_csv):
    print("⚠️ El archivo de predicciones ya existía y será sobrescrito.")

# Guardar el DataFrame completo en el archivo CSV
df_predicciones.to_csv(ruta_csv, index=False)

# Mensajes finales
print(f"✅ Se han generado predicciones semanales y se han guardado en '{ruta_csv}'")
print(f"📅 Total de predicciones generadas: {len(df_predicciones)}")
