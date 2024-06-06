import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns
from PIL import Image

# Ruta a la base de datos descomprimida
data_path = r'C:\Users\Henry\Downloads\orl_faces'

# Tamaño al que se reducirá cada imagen
resize_shape = (64, 64)

# Función para cargar imágenes y etiquetas
def cargar_imagenes_y_etiquetas(data_path, resize_shape):
    imagenes = []
    etiquetas = []
    for dir_name in os.listdir(data_path):
        if os.path.isdir(os.path.join(data_path, dir_name)):
            etiqueta = int(dir_name[1:]) - 1
            for file_name in os.listdir(os.path.join(data_path, dir_name)):
                file_path = os.path.join(data_path, dir_name, file_name)
                imagen = Image.open(file_path).convert('L').resize(resize_shape)
                imagenes.append(np.array(imagen))
                etiquetas.append(etiqueta)
    return np.array(imagenes), np.array(etiquetas)

# Función para calcular descriptores de textura
def calcular_descriptores_de_textura(imagen):
    descriptores = {}
    media = np.mean(imagen)
    desvio_std = np.std(imagen)

    # Cálculo de la entropía usando la fórmula proporcionada
    glcm = calcular_glcm(imagen)
    entropia = calcular_entropia(glcm)

    descriptores['media'] = media
    descriptores['desvio_std'] = desvio_std
    descriptores['entropia'] = entropia

    return descriptores

# Función para calcular la Matriz de Co-ocurrencia de Nivel de Gris (GLCM)
def calcular_glcm(imagen):
    niveles = 256
    glcm = np.zeros((niveles, niveles), dtype=float)
    filas, columnas = imagen.shape
    for i in range(filas - 1):
        for j in range(columnas - 1):
            fila_p = imagen[i, j]
            columna_p = imagen[i, j + 1]
            glcm[fila_p, columna_p] += 1

    glcm /= glcm.sum()
    return glcm

# Función para calcular la entropía a partir de la GLCM
def calcular_entropia(glcm):
    entropia = -np.sum(glcm * np.log2(glcm + (glcm == 0)))
    return entropia

# Función para normalizar los datos
def normalizar(X):
    media = np.mean(X, axis=0)
    desvio_std = np.std(X, axis=0)
    desvio_std[desvio_std == 0] = 1  # Evitar divisiones por cero
    return (X - media) / desvio_std, media, desvio_std

# Función para calcular PCA desde cero
def pca_manual(X, n_components):
    # Normalizar los datos
    X_norm, media, desvio_std = normalizar(X)
    print("Datos normalizados:", X_norm.shape)

    # Calcular la matriz de covarianza
    matriz_covarianza = np.cov(X_norm.T)
    print("Matriz de covarianza:", matriz_covarianza.shape)

    # Calcular los valores y vectores propios
    valores_propios, vectores_propios = np.linalg.eig(matriz_covarianza)

    # Convertir a reales
    valores_propios = np.real(valores_propios)
    vectores_propios = np.real(vectores_propios)

    # Ordenar los vectores propios por los valores propios más grandes
    idx = np.argsort(valores_propios)[::-1]
    vectores_propios = vectores_propios[:, idx]
    valores_propios = valores_propios[idx]

    # Seleccionar los primeros n componentes principales
    vectores_propios = vectores_propios[:, :n_components]

    # Transformar los datos
    X_reducido = np.dot(X_norm, vectores_propios)

    return X_reducido, valores_propios, vectores_propios, media, desvio_std

# Cargar imágenes y etiquetas
imagenes, etiquetas = cargar_imagenes_y_etiquetas(data_path, resize_shape)
print("Imágenes cargadas:", imagenes.shape)
print("Etiquetas cargadas:", etiquetas.shape)

# Vectorizar imágenes y calcular descriptores de textura
vectores_imagen = [img.flatten() for img in imagenes]
descriptores_textura = [list(calcular_descriptores_de_textura(img).values()) for img in imagenes]

print("Vectores de imagen:", len(vectores_imagen))
print("Descriptores de textura:", len(descriptores_textura))

# Convertir listas a arrays
vectores_imagen = np.array(vectores_imagen)
descriptores_textura = np.array(descriptores_textura)

print("Shape de vectores de imagen:", vectores_imagen.shape)
print("Shape de descriptores de textura:", descriptores_textura.shape)

# Combinar características de imagen y textura
caracteristicas_combinadas = np.hstack((vectores_imagen, descriptores_textura))
print("Shape de características combinadas:", caracteristicas_combinadas.shape)

# Aplicar PCA manualmente
n_components = 50  # Selecciona el número de componentes principales deseado
X_reducido, valores_propios, vectores_propios, media, desvio_std = pca_manual(caracteristicas_combinadas, n_components)

# Graficar Scree Plot
varianza_explicada = valores_propios / np.sum(valores_propios)
plt.figure(figsize=(10, 6))
plt.plot(np.cumsum(varianza_explicada))
plt.xlabel('Número de Componentes')
plt.ylabel('Varianza Explicada Acumulada')
plt.title('Scree Plot')
plt.grid()
plt.show()

# Dividir datos en entrenamiento y prueba (80%-20%)
X_entrenamiento, X_prueba, y_entrenamiento, y_prueba = train_test_split(X_reducido, etiquetas, test_size=0.2, random_state=42)

# Verificar tamaños de conjuntos de entrenamiento y prueba
print(f'Tamaño del conjunto de entrenamiento: {X_entrenamiento.shape[0]}')
print(f'Tamaño del conjunto de prueba: {X_prueba.shape[0]}')

# Asegurarse de que hay suficientes muestras para k-vecinos
if X_entrenamiento.shape[0] >= 3:
    # Clasificación utilizando k-vecinos más cercanos
    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(X_entrenamiento, y_entrenamiento)
    y_pred = knn.predict(X_prueba)

    # Evaluar el modelo
    precision = accuracy_score(y_prueba, y_pred)
    matriz_confusion = confusion_matrix(y_prueba, y_pred)

    print(f'Precisión: {precision:.2f}')

    # Graficar la matriz de confusión
    plt.figure(figsize=(10, 8))
    sns.heatmap(matriz_confusion, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicción')
    plt.ylabel('Actual')
    plt.title('Matriz de Confusión')
    plt.show()
else:
    print("No hay suficientes muestras en el conjunto de entrenamiento para k-vecinos.")
