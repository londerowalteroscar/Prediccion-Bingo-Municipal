import pandas as pd
import os

# Ruta al archivo Excel
ruta_archivo = os.path.join("data", "tombola.xlsx")

# Cargar el archivo
try:
    df = pd.read_excel(ruta_archivo)
except FileNotFoundError:
    print("❌ Error: No se encontró el archivo 'tombola.xlsx' en la carpeta 'data'.")
    exit()

# Eliminar filas sin número
df.dropna(subset=["Numero"], inplace=True)

# Calcular frecuencias
frecuencia = df["Numero"].value_counts().sort_values(ascending=False)

# Obtener top 10
top_10 = frecuencia.head(10).index.tolist()

# Mostrar la lista en formato requerido
print(top_10)
