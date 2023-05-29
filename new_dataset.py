import os
import cv2
from artificial_vision.image_recognition import ImageRecognition


class NewDataset:
    DATA_SET = 'dataset\\new'
    NEW_DATASET_FOLDER = 'dataset\\resized'

    def __init__(self) -> None:
        self.recognition = ImageRecognition("")

    def read_folder(self):
        content_dataset = os.listdir(self.DATA_SET)
        for image_folder in content_dataset:
            # images = os.path.join(self.ORIGINAL_DATASET_FOLDER, image_folder)
            path_image_folder = os.path.join(self.DATA_SET, image_folder)
            images = os.listdir(path_image_folder)
            for image_name in images:
                path_img = os.path.join(path_image_folder, image_name)
                img_read = cv2.imread(path_img)
                img_resized = cv2.resize(img_read,(128, 128))
                self._save_img_cropped(img_resized, image_folder,image_name)

    def _save_img_cropped(self, img_cropped, folder_to_save, image_name):
        path_folder = os.path.join(self.NEW_DATASET_FOLDER, folder_to_save)
        if not os.path.exists(path_folder):
            os.makedirs(path_folder)
        cv2.imwrite(os.path.join(path_folder, image_name), img_cropped)

if __name__ == "__main__":
    creator_new_dataset = NewDataset()
    creator_new_dataset.read_folder()