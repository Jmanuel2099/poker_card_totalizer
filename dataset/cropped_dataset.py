import os
import cv2
import numpy as np


class CroppedDataset:
    
    DATASET_FOLDER = 'dataset'
    TRAINING_DATASET = 'train' # training mode to save the image
    TEST_DATASET = 'test' # testing mode to save the image
    WIDTH_IMAGE = 128
    HEIGHT_IMAGE = 128
    CARDS_VALUES = [10, 7, 8, 9, 11, 12, 13] # -> [0, 7, 8, 9, j, k, q]

    def __init__(self) -> None:
        pass

    def save_img_cropped(self, img_cropped, folder_to_save, image_name):
        """
        save_img_cropped is in charge of saving the cropped image with its local name depending 
        on the mode (train or test) and returns the relative path where the cropped image was saved.
        """
        # path_folder = os.path.join(self.DATASET_FOLDER, self.TRAINING_DATASET, folder_to_save)
        path_folder = os.path.join(self.DATASET_FOLDER, self.TRAINING_DATASET, folder_to_save)
        if not os.path.exists(path_folder):
            os.makedirs(path_folder)
        path_image = os.path.join(path_folder, image_name)
        cv2.imwrite(path_image, img_cropped)
        return path_image

    def get_number_type_cards(self):
        """
        get_number_type_cards is in charge of getting the number of categories (7...k) to work with.
        """
        path_folder = os.path.join(self.DATASET_FOLDER, self.TRAINING_DATASET)
        card_types = os.listdir(path_folder)
        return len(card_types)

    def read_dataset(self, mode):
        """
        read_dataset is in charge of reading the data set to perform the flattening 
        of the images in the data set and provide the possible outputs needed for 
        the training of a convolutional network model. 
        """
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

    def get_card_value(self, probability):
        """
        get_card_value gets the card number in order to translate a probability into the value of the card.
        """
        return self.CARDS_VALUES[probability]
