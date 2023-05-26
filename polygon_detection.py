import cv2
import numpy as np


nameWindow="Calculadora"
def nothing(x):
    pass

def constructorVentana():
    cv2.namedWindow(nameWindow)
    cv2.createTrackbar("min",nameWindow,0,255,nothing)
    cv2.createTrackbar("max", nameWindow, 100, 255, nothing)
    cv2.createTrackbar("areaMin", nameWindow, 500, 10000, nothing)

def detectar_figura(img_original):

    resized_image = cv2.resize(img_original, (720, 480))

    img_gris = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
    cv2.imshow('Gris', img_gris)

    min=cv2.getTrackbarPos("min", nameWindow)
    max=cv2.getTrackbarPos("max", nameWindow)
    bordes=cv2.Canny(img_gris,min,max)
    tamañoKernel=cv2.getTrackbarPos("kernel", nameWindow)
    kernel=np.ones((tamañoKernel,tamañoKernel),np.uint8)
    bordes=cv2.dilate(bordes,kernel)
    cv2.imshow("Bordes_modificados", bordes)
    
    # Detección de la carta
    figuras,jerarquia=cv2.findContours(bordes,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    areas=calcular_areas(figuras)
    i=0
    areaMin=cv2.getTrackbarPos("areaMin", nameWindow)
    
    for i, contour in enumerate(figuras):
        if areas[i]>=areaMin:
            # Crear una imagen en blanco del mismo tamaño que la imagen original
            contour_image = np.zeros_like(resized_image)

            # Dibujar el contorno en la imagen en blanco
            cv2.drawContours(contour_image, [contour], -1, (255, 255, 255), cv2.FILLED)

            # Recortar la región dentro del contorno en la imagen original
            x, y, w, h = cv2.boundingRect(contour)
            cropped = resized_image[y:y+h, x:x+w]
        i=i+1
    return cropped

def calcular_areas(figuras):
    areas=[]
    for figuraActual in figuras:
        areas.append(cv2.contourArea(figuraActual))
    return areas

video = cv2.VideoCapture(1)
constructorVentana()
while True:
    _, frame = video.read()
    detectar_figura(frame)
    cv2.imshow('Imagen', frame)

    key = cv2.waitKey(5) & 0xFF
    if key == 112:
        break

    if key == 27:
        break

    

video.release()
cv2.destroyAllWindows()


# import cv2

# # Leer la imagen en escala de grises
# image = cv2.imread('7C18.jpg')
# resized_image = cv2.resize(image, (720, 480))

# # Aplicar Canny para detectar los bordes
# edges = cv2.Canny(resized_image, 100, 200)

# # Mostrar la imagen original y la imagen de bordes
# cv2.imshow('Original', resized_image)
# cv2.imshow('Bordes', edges)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


# import cv2

# # Cargar la imagen
# image = cv2.imread('7C18.jpg')
# resized_image = cv2.resize(image, (720, 480))

# # Convertir la imagen a escala de grises
# gray = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)

# # Aplicar umbralización adaptativa
# thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)

# # Mostrar la imagen original y la imagen umbralizada
# cv2.imshow('Original', resized_image)
# cv2.imshow('Umbralizada', thresh)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
