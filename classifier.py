import tensorflow as tf
import numpy as np
from PIL import Image


class TensorflowLiteClassificationModel:
    def __init__(self, model_path, labels, image_size):
        self.interpreter = tf.lite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        self._input_details = self.interpreter.get_input_details()
        self._output_details = self.interpreter.get_output_details()
        self.labels = labels
        self.image_size = image_size

    def run(self, image):
        self.interpreter.set_tensor(self._input_details[0]["index"], image)
        self.interpreter.invoke()
        tflite_interpreter_output = self.interpreter.get_tensor(self._output_details[0]["index"])
        probabilities = np.array(tflite_interpreter_output[0])

        label_to_probabilities = []
        for i, probability in enumerate(probabilities):
            label_to_probabilities.append([self.labels[i], float(probability)])
        return sorted(label_to_probabilities, key=lambda element: element[1])

    def run_from_filepath(self, image_path):
        input_data_type = self._input_details[0]["dtype"]
        data = Image.open(image_path)
        image = np.array(data)
        if input_data_type == np.float32:
            image = image / 255.
        if image.shape == (1, 180, 180):
            image = np.stack(image * 3, axis=0)
        return self.run(image)

