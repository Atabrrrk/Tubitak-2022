import tensorflow as tf
from keras import models, layers, regularizers
from keras.preprocessing.image import ImageDataGenerator
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

image_height = 180
image_width = 180
batch_size = 16
steps_for_valid = 20
epoch_size = 32
biome_train_path = r"C:\Users\pc\PycharmProjects\Tubitak\Training\Train"
biome_validation_path = r"C:\Users\pc\PycharmProjects\Tubitak\Training\Validation"
biome_test_path = r"C:\Users\pc\PycharmProjects\Tubitak\Training\Test"

biome_classifier = models.Sequential()
biome_classifier.add(layers.Conv2D(16, (3, 3), activation='relu', input_shape=(180, 180, 1)))
biome_classifier.add(layers.MaxPooling2D((2, 2)))
biome_classifier.add(layers.Conv2D(32, (3, 3), activation='relu'))
biome_classifier.add(layers.MaxPooling2D((2, 2)))
biome_classifier.add(layers.Flatten())
biome_classifier.add(layers.Dense(32, activation="relu"))
biome_classifier.add(layers.Dense(8, activation="relu"))
biome_classifier.add(layers.Dense(2, activation="relu"))

datagen1 = ImageDataGenerator(
    rescale=1.0 / 255.0,
    rotation_range=25,
    zoom_range=(0.7, 0.7),
    horizontal_flip=False,
    vertical_flip=True,
    fill_mode="reflect",
    data_format="channels_last",
    brightness_range=[0.2, 1.2],
)

datagen2 = ImageDataGenerator(rescale=1.0 / 255.0)

biome_train_generator = datagen1.flow_from_directory(
    directory=biome_train_path,
    target_size=(image_height, image_width),
    color_mode="grayscale",
    batch_size=batch_size,
    class_mode="sparse",
    shuffle=False,
    subset="training",
    seed=128,
)

biome_validation_generator = datagen2.flow_from_directory(
    directory=biome_validation_path,
    target_size=(image_height, image_width),
    color_mode="grayscale",
    batch_size=batch_size,
    class_mode="sparse",
    shuffle=False,
    subset="validation",
    seed=128,
)

biome_test_generator = datagen2.flow_from_directory(
    directory=biome_test_path,
    target_size=(image_height, image_width),
    color_mode="grayscale",
    batch_size=batch_size,
    class_mode="sparse",
    shuffle=False,
    seed=128,
)

biome_classifier.compile(optimizer="adam",
                         loss="sparse_categorical_crossentropy",
                         metrics=["accuracy"])

biome_classifier.fit(biome_train_generator, epochs=epoch_size, steps_per_epoch=50,
                     validation_data=biome_validation_generator, validation_steps=30
                     )
biome_classifier.evaluate(biome_test_generator, steps=steps_for_valid, verbose=1)
biome_classifier("biome_classify_model.h5")
tf_lite_biome_converter = tf.lite.TFLiteConverter.from_keras_model(biome_classifier)
tflite_biome_model = tf_lite_biome_converter.convert()
open("tflite_biome_classify_model.tflite", "wb").write(tflite_biome_model)
