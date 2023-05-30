from keras.models import Sequential
from keras.layers import InputLayer,Input,Conv2D, MaxPool2D,Reshape,Dense,Flatten
from keras.models import load_model
from dataset.cropped_dataset import CroppedDataset
import numpy as np
import cv2
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, confusion_matrix
from keras.losses import categorical_crossentropy


class ModelTwo:
    def __init__(self) -> None:
        self.dataset = CroppedDataset()
        self.model = Sequential()
        self.path_to_save_model = 'deep_learning\models\model_two\model_two.h5'
        self.trained_model = None
        self.loss = ''
        self.accuracy = ''
        self.f1_score = ''
        self.precision = ''
        self.recall = ''
        self.correlation_matrix = ''

    def get_model(self):
        return self.model

    def run(self):
        """
        run is the main method and does the whole process from creating it to 
        testing the deep learning model.
        """
        self._create_model()
        self.train_model(epochs=50, batch_size=60)
        self._save_model()
        self._test_model()
        print(f'accuracy: {self.accuracy}, loss: {self.loss}, f1: {self.f1_score}, precision: {self.precision}, recall: {self.recall}')
        print('Matrix de correlacion')
        print(self.correlation_matrix)

    def predict(self, path_image):
        """
        predict is in charge of receiving an image and then adapting it so that the deep learning 
        model can make a prediction and say which category of card is more probable.
        """
        self._load_model_trained()
        image = cv2.imread(path_image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = cv2.resize(image, (self.dataset.WIDTH_IMAGE, self.dataset.HEIGHT_IMAGE))
        image = image.flatten()
        image = image / 255

        loaded_images = []
        loaded_images.append(image)
        loaded_images_npa = np.array(loaded_images)
        prediction = self.trained_modelc.predict(x=loaded_images_npa)
        print("Prediction", prediction)
        major_classes = np.argmax(prediction, axis=1)
        print(major_classes)
        return major_classes[0]

    def _create_model(self):
        """
        _create_model configures and creates the deep learning model to be trained 
        and used for card predictions.
        """
        self._create_input_layer()
        self._create_convolutional_layer(kernel=1,
                                        strides=1,
                                        filters=36,
                                        padding="same",
                                        activation="relu",
                                        layer_name="layer_3",
                                        pooling=2,
                                        strides_pooling=2)
        self._create_convolutional_layer(kernel=3,
                                        strides=1,
                                        filters=128,
                                        padding="same",
                                        activation="relu",
                                        layer_name="layer_2",
                                        pooling=2,
                                        strides_pooling=2)
        self._create_convolutional_layer(kernel=5, 
                                        strides=2, 
                                        filters=144,
                                        padding="same",
                                        activation="relu",
                                        layer_name="layer_4",
                                        pooling=2,
                                        strides_pooling=2)
        self._create_convolutional_layer(kernel=5, 
                                        strides=2, 
                                        filters=256,
                                        padding="same",
                                        activation="relu",
                                        layer_name="layer_1",
                                        pooling=2,
                                        strides_pooling=2)
        
        self._flatten(activation="relu")
        self._create_output_layer(activation="softmax")
        self._translate_keras_to_tensorflow()

    def _define_image_shape(self):
        """
        _define_image_shape defines the dimensions of the images that will interact 
        in the prediction and training of the neural network
        """
        number_chanels = 1 # esta en escala de grises 
        image_shape = (self.dataset.WIDTH_IMAGE, self.dataset.HEIGHT_IMAGE, number_chanels)
        pixels = self.dataset.WIDTH_IMAGE * self.dataset.HEIGHT_IMAGE
        return image_shape, pixels

    def _create_input_layer(self):
        """
        _create_input_layer is in charge of configuring and adding the input layer of the neural network.
        """
        image_shape, pixels = self._define_image_shape()
        self.model.add(InputLayer(input_shape=(pixels,)))
        self.model.add(Reshape(image_shape))

    def _create_convolutional_layer(self, 
                                    kernel, 
                                    strides, 
                                    filters, 
                                    padding, 
                                    activation, 
                                    layer_name, 
                                    pooling, 
                                    strides_pooling):
        """
        _create_convolutional_layer is in charge of creating, configuring 
        and adding the convolutional layers to the deep learning model.
        """
        self.model.add(Conv2D(kernel_size=kernel,
                              strides=strides,
                              filters=filters,
                              padding=padding,
                              activation=activation,
                              name=layer_name))
        self.model.add(MaxPool2D(pool_size=pooling,strides=strides_pooling))

    def _create_output_layer(self, activation):
        """
        _create_output_layer is in charge of creating the output layer of the neural 
        network model with a given activation function.
        """
        self.model.add(Dense(self.dataset.get_number_type_cards(), activation=activation))

    def _flatten(self, activation):
        """
        _flatten is in charge of creating the penultimate layer of the network which is 
        to be flattened to give an output to the next layer.
        """
        self.model.add(Flatten())
        self.model.add(Dense(128, activation=activation))

    def _translate_keras_to_tensorflow(self):
        """
        _translate_keras_to_tensorflow is in charge of translating a model made with the keras 
        library to the tensorflow library.
        """
        self.model.compile(optimizer="adam",loss="categorical_crossentropy", metrics=["accuracy"])

    def train_model(self, epochs, batch_size):
        """
        train_model from a configured model is in charge of training it with the training dataset in a 
        determined number of epochs and with a given batch size.
        """
        train_images, train_probabilities = self.dataset.read_dataset(self.dataset.TRAINING_DATASET)
        self.model.fit(x=train_images,
                       y=train_probabilities,
                       epochs=epochs,
                       batch_size=batch_size)

    def _test_model(self):
        """
        _test_model from an already trained model obtains the metrics of 
        the model to measure its accuracy in prediction
        """
        test_images, test_probabilities = self.dataset.read_dataset(self.dataset.TEST_DATASET)
        results = self.model.evaluate(x=test_images, y=test_probabilities)
        self.accuracy = results[1]
        self.loss = results[0]

        results = self.trained_model.predict(test_images)
        y_predict = np.argmax(results, axis=1)
        y_true = np.argmax(test_probabilities, axis=1)
        self.precision = precision_score(y_true, y_predict, average='weighted')
        self.recall = recall_score(y_true, y_predict, average='weighted')
        self.f1_score = f1_score(y_true, y_predict, average='weighted')
        self.correlation_matrix=confusion_matrix(y_true, y_predict)

    def _save_model(self):
        """
        _save_model saves the model locally with the .h5 extension.
        """
        self.model.save(self.path_to_save_model)
        self._load_model_trained()

    def _load_model_trained(self):
        """
        _load_model_trained is in charge of loading the trained model from 
        the .h5 file to be used to obtain tests or predictions.
        """
        self.trained_model = load_model(self.path_to_save_model)