import cv2
import numpy as np


class ImageRecognition:
    def __init__(self, window) -> None:
        self.name_window = window

    def crop(self, gray_image, contours):
        """
        crop from a grayscale image and its contours is responsible for 
        recognizing the figures in a grayscale image to keep only the important information of the image. 
        """
        areas= self._calcular_areas(contours)
        i = 0
        areaMin = cv2.getTrackbarPos("areaMin", self.name_window)
        # areaMin = 7007
        rois = []
        for contour in contours:
            if areas[i]>=areaMin:
                # Crear una imagen en blanco del mismo tamaño que la imagen original
                contour_image = np.zeros_like(gray_image)
                # Dibujar el contorno en la imagen en blanco
                cv2.drawContours(contour_image, [contour], -1, (255, 255, 255), cv2.FILLED)
                # Recortar la región dentro del contorno en la imagen original
                x, y, w, h = cv2.boundingRect(contour)
                cropped = gray_image[y:y+h, x:x+w]
                rois.append(cropped)
            i=i+1
        return rois

    def detect_figure_from_video(self, image_original):
        """
        detect_figure_from_video is responsible for detecting the contours of the figures in a 
        grayscale video from the minimum and minimum(trackbars), and then used 
        in the recognition of the figure and subsequent cropping.
        """
        imgame_gris = cv2.cvtColor(image_original, cv2.COLOR_BGR2GRAY) # convertir a escala de grises
        min=cv2.getTrackbarPos("min", self.name_window)
        max=cv2.getTrackbarPos("max", self.name_window)
        binary_image=cv2.Canny(imgame_gris,min,max) # convertir a imagen binaria
        contours, _=cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for figuraActual in contours:
            cv2.drawContours(image_original, [figuraActual], 0, (0, 0, 255), 2)
        return imgame_gris, contours

    def _calcular_areas(self, figuras):
        """
        _calculate_areas is in charge of calculating the areas from the contours 
        of the figures detected in an image and then use them for proper recognition of the figures. 
        """
        areas=[]
        for figuraActual in figuras:
            areas.append(cv2.contourArea(figuraActual))
        return areas
