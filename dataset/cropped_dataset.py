import os
import cv2
import numpy as np


class CroppedDataset:
    DATASET_FOLDER = 'C:\\Users\\jmanu\\PersonalProjects\\poker_card_totalizer\\dataset'
    TRAINING_DATASET = 'train'
    TEST_DATASET = 'test'
    WIDTH_IMAGE = 128
    HEIGHT_IMAGE = 128
    CARDS_VALUES = [10, 7, 8, 9, 11, 12, 13] # -> [0, 7, 8, 9, 11, 12, 13]

    def __init__(self) -> None:
        pass

    def save_img_cropped(self, img_cropped, folder_to_save, image_name):
        # path_folder = os.path.join(self.DATASET_FOLDER, self.TRAINING_DATASET, folder_to_save)
        path_folder = os.path.join(self.DATASET_FOLDER, self.TRAINING_DATASET, folder_to_save)
        if not os.path.exists(path_folder):
            os.makedirs(path_folder)
        path_image = os.path.join(path_folder, image_name)
        cv2.imwrite(path_image, img_cropped)
        return path_image

    def get_number_type_cards(self):
        path_folder = os.path.join(self.DATASET_FOLDER, self.TRAINING_DATASET)
        card_types = os.listdir(path_folder)
        return len(card_types)

    def read_dataset(self, mode):
        loaded_images = []
        expected_card = []
        path_folder = os.path.join(self.DATASET_FOLDER, mode)
        card_types = os.listdir(path_folder)
        for i, card_type in enumerate(card_types):
            path_cards = os.path.join(path_folder, card_type)
            cards = os.listdir(path_cards)
            for card in cards:
                path_card = os.path.join(path_cards, card)
                print(path_card)
                image = cv2.imread(path_card)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                image = cv2.resize(image, (self.WIDTH_IMAGE, self.HEIGHT_IMAGE))
                image = image.flatten()
                image = image / 255
                loaded_images.append(image)
                probabilities = np.zeros(len(card_types))
                probabilities[i] = 1
                expected_card.append(probabilities)
        training_images = np.array(loaded_images)
        expected_cards = np.array(expected_card)
        return training_images, expected_cards

    # def _convert_cart_type_to_binary(self, card_type):
    #     ascii_value = ord(card_type)
    #     binary_value = format(ascii_value, '0' + str('7') + 'b')
    #     binary_list = list(binary_value)

    #     return list(map(int, binary_list))
    
    def get_card_value(self, value):
        return self.CARDS_VALUES[value]

    # def _convert_binary_to_cart_type(self, binary_arr):
    #     binary = int(''.join(str(element) for element in binary_arr))
    #     ascii_value = int(binary, 2)
    #     return chr(ascii_value)
