import os
import cv2
from artificial_vision.image_recognition import ImageRecognition


class NewDataset:
    NEW_DATASET_FOLDER = 'C:\\Users\\jmanu\\OneDrive\\Documents\\Semestre\\InteligentesII\\DeepLearning\\SegundoParcial\\code_second_partial\\dataset\\cropped'
    ORIGINAL_DATASET_FOLDER = 'C:\\Users\\jmanu\\OneDrive\\Documents\\Semestre\\InteligentesII\\DeepLearning\\SegundoParcial\\code_second_partial\\dataset\\original'

    def __init__(self) -> None:
        self.recognition = ImageRecognition("")

    def read_folder(self):
        content_dataset = os.listdir(self.ORIGINAL_DATASET_FOLDER)
        for image_folder in content_dataset:
            # images = os.path.join(self.ORIGINAL_DATASET_FOLDER, image_folder)
            path_image_folder = os.path.join(self.ORIGINAL_DATASET_FOLDER, image_folder)
            images = os.listdir(path_image_folder)
            for image_name in images:
                path_img = os.path.join(path_image_folder, image_name)
                gray_image, countours = self.recognition.detect_figure_from_file(path_img)
                image_cropped = self.recognition.crop(gray_image, countours)
                self._save_img_cropped(image_cropped, image_folder, image_name)

    def _save_img_cropped(self, img_cropped, folder_to_save, image_name):
        path_folder = os.path.join(self.NEW_DATASET_FOLDER, folder_to_save)
        if not os.path.exists(path_folder):
            os.makedirs(path_folder)
        cv2.imwrite(os.path.join(path_folder, image_name), img_cropped)

if __name__ == "__main__":
    creator_new_dataset = NewDataset()
    creator_new_dataset.read_folder()