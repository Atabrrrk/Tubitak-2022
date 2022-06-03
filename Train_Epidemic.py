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
epidemic_train_path = r"C:\Users\pc\PycharmProjects\Tubitak\Training\Train_Epidemic"
epidemic_validation_path = r"C:\Users\pc\PycharmProjects\Tubitak\Training\Validation_Epidemic"
epidemic_test_path = r"C:\Users\pc\PycharmProjects\Tubitak\Training\Test_Epidemic"

epidemic_classifier = models.Sequential()
epidemic_classifier.add(layers.Conv2D(16, (3, 3), activation='relu', input_shape=(180, 180, 1)))
epidemic_classifier.add(layers.MaxPooling2D((2, 2)))
epidemic_classifier.add(layers.Conv2D(32, (3, 3), activation='relu'))
epidemic_classifier.add(layers.MaxPooling2D((2, 2)))
epidemic_classifier.add(layers.Flatten())
epidemic_classifier.add(layers.Dense(32, activation="relu"))
epidemic_classifier.add(layers.Dense(8, activation="relu"))
epidemic_classifier.add(layers.Dense(2, activation="relu"))

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

epidemic_train_generator = datagen1.flow_from_directory(
    directory=epidemic_train_path,
    target_size=(image_height, image_width),
    color_mode="rgb",
    batch_size=batch_size,
    class_mode="sparse",
    shuffle=False,
    subset="training",
    seed=128,
)

epidemic_validation_generator = datagen2.flow_from_directory(
    directory=epidemic_validation_path,
    target_size=(image_height, image_width),
    color_mode="rgb",
    batch_size=batch_size,
    class_mode="sparse",
    shuffle=False,
    subset="validation",
    seed=128,
)

epidemic_test_generator = datagen2.flow_from_directory(
    directory=epidemic_test_path,
    target_size=(image_height, image_width),
    color_mode="rgb",
    batch_size=batch_size,
    class_mode="sparse",
    shuffle=False,
    seed=128
)

epidemic_classifier.compile(optimizer="adam",
                            loss="sparse_categorical_crossentropy",
                            metrics=["accuracy"])

epidemic_classifier.fit(epidemic_train_generator, epochs=epoch_size, steps_per_epoch=20,
                        validation_data=epidemic_validation_generator, validation_steps=20
                        )

epidemic_classifier.evaluate(directory=epidemic_test_generator, steps=steps_for_valid, verbose=1)
epidemic_classifier("epidemic_classify_model.h5")
tf_lite_epidemic_converter = tf.lite.TFLiteConverter.from_keras_model(epidemic_classifier)
tflite_epidemic_model = tf_lite_epidemic_converter.convert()
open("tflite_epidemic_classify_model.tflite", "wb").write(tflite_epidemic_model)

--------------------------------------------------------------------------------
