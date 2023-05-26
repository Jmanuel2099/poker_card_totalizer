import cv2
import numpy as np


# Cargar la imagen
# img = cv2.imread("7C41.jpg")
resized_image = cv2.imread("7C41.jpg")
# resized_image = cv2.resize(img, (720, 480))

# Convertir la imagen a escala de grises
gray = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)

# Aplicar un umbral para obtener una imagen binaria
_, threshold = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

# Encontrar contornos en la imagen binaria
contours, _ = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Iterar sobre cada contorno y crear una imagen para cada uno
for i, contour in enumerate(contours):
    # Crear una imagen en blanco del mismo tamaño que la imagen original
    contour_image = np.zeros_like(resized_image)

    # Dibujar el contorno en la imagen en blanco
    cv2.drawContours(contour_image, [contour], -1, (255, 255, 255), cv2.FILLED)

    # Recortar la región dentro del contorno en la imagen original
    x, y, w, h = cv2.boundingRect(contour)
    cropped = resized_image[y:y+h, x:x+w]

    # Guardar la imagen recortada para el contorno actual
    cv2.imwrite(f"contorno_{i}.jpg", cropped)
