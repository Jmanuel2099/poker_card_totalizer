from keras.models import Sequential
from keras.layers import InputLayer,Input,Conv2D, MaxPool2D,Reshape,Dense,Flatten
from dataset.cropped_dataset import CroppedDataset


class ModelOne:
    def __init__(self) -> None:
        self.dataset = CroppedDataset()
        self.model = Sequential()
        self.path_to_save_model = 'deep_learning\models\model_one\model_one.h5'

    def get_model(self):
        return self.model

    def run(self):
        self._create_model()
        self.train_model()
        self.test_model()
        self.save_model()

    def _create_model(self):
        self._create_input_layer()
        self._create_convolutional_layer("layer_1")
        self._create_convolutional_layer("layer_2")
        self._flatten()
        self._create_output_layer()
        self._translate_keras_to_tensorflow()

    def _define_image_shape(self):
        number_chanels = 1 # esta en escala de grises 
        image_shape = (self.dataset.WIDTH_IMAGE, self.dataset.HEIGHT_IMAGE, number_chanels)
        pixels = self.dataset.WIDTH_IMAGE * self.dataset.HEIGHT_IMAGE
        return image_shape, pixels

    def _create_input_layer(self):
        image_shape, pixels = self._define_image_shape()
        self.model.add(InputLayer(input_shape=(pixels,)))
        self.model.add(Reshape(image_shape))

    def _create_convolutional_layer(self, layer_name):
        self.model.add(Conv2D(kernel_size=5,strides=2,filters=16,padding="same",activation="relu",name=layer_name))
        self.model.add(MaxPool2D(pool_size=2,strides=2))

    def _create_output_layer(self):
        self.model.add(Dense(self.dataset.get_number_type_cards(), activation="softmax"))

    def _flatten(self):
        self.model.add(Flatten())
        self.model.add(Dense(128,activation="relu"))

    def _translate_keras_to_tensorflow(self):
        self.model.compile(optimizer="adam",loss="categorical_crossentropy", metrics=["accuracy"])

    def train_model(self):
        train_images, train_probabilities = self.dataset.read_dataset(self.dataset.TRAINING_DATASET)
        self.model.fit(x=train_images, y=train_probabilities, epochs=30, batch_size=60)

    def test_model(self):
        test_images, test_probabilities = self.dataset.read_dataset(self.dataset.TEST_DATASET)
        results = self.model.evaluate(x=test_images, y=test_probabilities)
        return results[1] # retorno el accuracy

    def save_model(self):
        self.model.save(self.path_to_save_model)
