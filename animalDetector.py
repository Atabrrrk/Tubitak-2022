import keras
import tensorflow as tf
from keras import models, layers, regularizers
from keras.preprocessing.image import ImageDataGenerator
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

image_height = 180
image_width = 180
batch_size = 32
steps_for_valid = 20
epoch_size = 15
animal_train_path = r"C:\Users\pc\PycharmProjects\Tubitak\Training\Train_Epidemic"
animal_validation_path = r"C:\Users\pc\PycharmProjects\Tubitak\Training\Validation_Epidemic"
animal_test_path = r"C:\Users\pc\PycharmProjects\Tubitak\Training\Test_Epidemic"

animal_classifier = models.Sequential()
animal_classifier.add(layers.Conv2D(16, (3, 3), activation='relu', input_shape=(180, 180, 1)))
animal_classifier.add(layers.MaxPooling2D((2, 2)))
animal_classifier.add(layers.Conv2D(32, (3, 3), activation='relu'))
animal_classifier.add(layers.MaxPooling2D((2, 2)))
animal_classifier.add(layers.Flatten())
animal_classifier.add(layers.Dense(32, activation="relu"))
animal_classifier.add(layers.Dense(8, activation="relu"))
animal_classifier.add(layers.Dense(2, activation="relu"))

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

animal_train_generator = datagen1.flow_from_directory(
    directory=animal_train_path,
    target_size=(image_height, image_width),
    color_mode="rgb",
    batch_size=batch_size,
    class_mode="sparse",
    shuffle=False,
    subset="training",
    seed=128,
)

animal_validation_generator = datagen2.flow_from_directory(
    directory=animal_validation_path,
    target_size=(image_height, image_width),
    color_mode="rgb",
    batch_size=batch_size,
    class_mode="sparse",
    shuffle=False,
    subset="validation",
    seed=128,
)

animal_test_generator = datagen2.flow_from_directory(
    directory=animal_test_path,
    target_size=(image_height, image_width),
    color_mode="rgb",
    batch_size=batch_size,
    class_mode="sparse",
    shuffle=False,
    seed=128
)

animal_classifier.compile(optimizer="adam",
                          loss="sparse_categorical_crossentropy",
                          metrics=["accuracy"])

animal_classifier.fit(animal_train_generator, epochs=epoch_size, steps_per_epoch=20,
                      validation_data=animal_validation_generator, validation_steps=20
                      )

animal_classifier.evaluate(directory=animal_test_generator, steps=steps_for_valid, verbose=1)
animal_classifier("epidemic_classify_model.h5")
tf_lite_animal_converter = tf.lite.TFLiteConverter.from_keras_model(animal_classifier)
tflite_animal_model = tf_lite_animal_converter.convert()
open("tflite_epidemic_classify_model.tflite", "wb").write(tflite_animal_model)
