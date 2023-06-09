import cv2
import numpy as np
from artificial_vision.image_recognition import ImageRecognition
from dataset.cropped_dataset import CroppedDataset
# from deep_learning.models.model_one.model_one import ModelOne
# from deep_learning.models.model_two.model_two import ModelTwo
from deep_learning.models.model_three.model_three import ModelThree
import datetime


class Window:
    OPTION_VIDEO_CAMERA = 0 # 1-> Web cam, 0->Phone
    KEYS_TAKE_PICTURE = [55, 56, 57, 48, 107, 106, 113] # [7, 8, 9, 0, k, j, q]

    def __init__(self) -> None:
        self.name_window = 'poker_card_totalizer'
        self.image_recognition = ImageRecognition(self.name_window)
        self.dataset = CroppedDataset()

    def _create_window(self):
        """
        _create_window is in charge of creating a window for the user to interact and
        configure several parameters with the help of trackbar.
        """
        cv2.namedWindow(self.name_window)
        cv2.createTrackbar("min", self.name_window, 0, 255, self._nothing)
        cv2.createTrackbar("max",  self.name_window, 100, 255, self._nothing)
        cv2.createTrackbar("kernel", self.name_window, 1, 100, self._nothing)
        cv2.createTrackbar("areaMin", self.name_window, 500, 10000, self._nothing)

    def _new_video_capture(self):
        """
        _new_video_capture is in charge of creating an instance of VideoCapture 
        to generate a video capture window with it.
        """
        return cv2.VideoCapture(self.OPTION_VIDEO_CAMERA)

    def run_window(self):
        """
        run_window is in charge of initializing the video capture and setting the different 
        options that the application has ('esc' -> exit the application, 
        'p' -> take picture to recognize the cards, 
        '7','8','9','0','j','k' or 'q' -> take picture for the dataset). 
        """
        cumulated = 0
        sum = 0
        self._create_window()
        video = self._new_video_capture()
        count_image = 61 # se utiliza para nombrar las imagenes con un consecutivo
        while True:
            _, frame = video.read()
            imgame_gris, contours = self.image_recognition.detect_figure_from_video(frame)
            cv2.putText(frame, f'Acumulado {cumulated}, Sum {sum}', (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.imshow("Window", frame)

            key = cv2.waitKey(5) & 0xFF
            if key == 27:
                break
            if key == 112:
                sum = 0
                start_time = datetime.datetime.now() # tiempo en el que incia el proceso de prediccion.
                model_to_predict = ModelThree()
                images_cropped = self.image_recognition.crop(imgame_gris, contours)
                for i, img in enumerate(images_cropped):
                    path_img = self.dataset.save_img_cropped(img, chr(key), f'{chr(key)}C{count_image}_{i}.jpg')
                    prediction = model_to_predict.predict(path_img)
                    value = self.dataset.get_card_value(prediction)
                    sum = sum + value
                cumulated = cumulated + sum
                end_time = datetime.datetime.now() # termina el tiempo de prediccion.
                print("Time: ",end_time - start_time)
            if key in self.KEYS_TAKE_PICTURE:
                images_cropped = self.image_recognition.crop(imgame_gris, contours)
                for image_cropped in images_cropped:
                    self.dataset.save_img_cropped(image_cropped, chr(key), f'{chr(key)}C{count_image}.jpg')
                count_image = count_image + 1

    def _nothing(x):
        pass
