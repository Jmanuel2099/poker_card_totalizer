import cv2
import numpy as np
from artificial_vision.image_recognition import ImageRecognition
from dataset.cropped_dataset import CroppedDataset


class Window:
    OPTION_VIDEO_CAMERA = 0 # 1-> Web cam, 0->Phone
    KEYS_TAKE_PICTURE = [55, 56, 57, 48, 107, 106, 113] # [7, 8, 9, 0, k, j, q]

    def __init__(self) -> None:
        self.name_window = 'Nombre'
        self.image_recognition = ImageRecognition(self.name_window)
        self.dataset_creator = CroppedDataset()


    def _create_window(self):
        cv2.namedWindow(self.name_window)
        cv2.createTrackbar("min", self.name_window, 0, 255, self._nothing)
        cv2.createTrackbar("max",  self.name_window, 100, 255, self._nothing)
        cv2.createTrackbar("kernel", self.name_window, 1, 100, self._nothing)
        cv2.createTrackbar("areaMin", self.name_window, 500, 10000, self._nothing)

    def _new_video_capture(self):
        return cv2.VideoCapture(self.OPTION_VIDEO_CAMERA)

    def run_window(self):
        self._create_window()
        video = self._new_video_capture()
        i = 61 # se utiliza para nombrar las imagenes con un consecutivo
        while True:
            _, frame = video.read()
            imgame_gris, contours = self.image_recognition.detect_figure_from_video(frame)
            cv2.imshow("Window", frame)

            key = cv2.waitKey(5) & 0xFF
            if key == 27:
                break
            if key == 112:
                self.image_recognition.crop(imgame_gris, contours)
            if key in self.KEYS_TAKE_PICTURE:
                image_cropped = self.image_recognition.crop(imgame_gris, contours)
                self.dataset_creator.save_img_cropped(image_cropped, chr(key), f'{chr(key)}C{i}.jpg')
                i = i + 1

    def _nothing(x):
        pass
