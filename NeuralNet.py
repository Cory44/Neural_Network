import numpy as np


class NeuralNet:

    def __init__(self, inputs, outputs):
        self._inputs = inputs
        self._outputs = outputs
        self.trained_iterations = 0

        self.input_size = len(self._inputs[0])
        self.output_size = len(self._outputs[0])

        self._hidden_weights = 2 * np.random.random((self.input_size, 128)) - 1
        self._hidden_weights2 = 2 * np.random.random((128, 16)) - 1
        self._output_weights = 2 * np.random.random((16, self.output_size)) - 1

    def train(self, iterations):
        trained_outputs = self._outputs

        for i in range(iterations):
            # Times input by weights, then normalize using sigmoid
            # on both hidden and output layers
            trained_hidden1 = self.sigmoid(np.dot(self._inputs, self._hidden_weights))
            trained_hidden2 = self.sigmoid(np.dot(trained_hidden1, self._hidden_weights2))
            trained_outputs = self.sigmoid(np.dot(trained_hidden2, self._output_weights))

            # Caluclate error and adjusment on output layer
            o_error = trained_outputs - self._outputs
            o_adjustments = o_error * self.sigmoid_derivative(trained_outputs)

            # Calculate error and adjustment on hidden layers
            h_error2 = np.dot(o_adjustments, self._output_weights.T)
            h_adjustments2 = h_error2 * self.sigmoid_derivative(trained_hidden2)

            h_error = np.dot(h_error2, self._hidden_weights2.T)
            h_adjustments1 = h_error * self.sigmoid_derivative(trained_hidden1)

            # Back propogate and adjust weights
            w1 = np.dot(self._inputs.T, h_adjustments1)
            w2 = np.dot(trained_hidden1.T, h_adjustments2)
            w3 = np.dot(trained_hidden2.T, o_adjustments)

            self._hidden_weights -= 0.0001 * w1
            self._hidden_weights2 -= 0.0001 * w2
            self._output_weights -= 0.0001 * w3
            print(i)

        self.trained_iterations = iterations

        return trained_outputs

    @staticmethod
    def sigmoid(x):
        a = 1 / (1 + np.exp(-x))
        return a

    @staticmethod
    def sigmoid_derivative(x):
        return x * (1 - x)

    def classify(self, image):
        classified_hidden = self.sigmoid(np.dot(image, self._hidden_weights))
        classified_hidden2 = self.sigmoid(np.dot(classified_hidden, self._hidden_weights2))
        classified_output = np.around(self.sigmoid(np.dot(classified_hidden2, self._output_weights)), decimals=2)

        return classified_output

    def bulk_classify(self, images):
        classified_outputs = []

        for image in images:
            classified_outputs.append(self.classify(image))

        return classified_outputs
