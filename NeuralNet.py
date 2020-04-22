import numpy as np

class NeuralNet:

    def __init__(self, layers, inputs, outputs):
        self.layers = layers
        self.inputs = inputs
        self.outputs = outputs

        self.input_size = len(self.inputs[0])

        self.hidden_weights = 2 * np.random.random((self.input_size, 128)) - 1
        self.hidden_weights2 = 2 * np.random.random((128, 16)) - 1
        self.output_weights = 2 * np.random.random((16, 10)) - 1

    def train(self, n):
        trained_outputs = self.outputs

        for i in range (n):
            # Times input by weights, then normalize using sigmoid
            # on both hidden and output layers
            trained_hidden1 = self.sigmoid(np.dot(self.inputs, self.hidden_weights))
            trained_hidden2 = self.sigmoid(np.dot(trained_hidden1, self.hidden_weights2))
            trained_outputs = self.sigmoid(np.dot(trained_hidden2, self.output_weights))

            # Caluclate error and adjusment on output layer
            o_error = trained_outputs - self.outputs
            o_adjustments = o_error * self.sigmoid_derivative(trained_outputs)

            # Calculate error and adjustment on hidden layers
            h_error2 = np.dot(o_adjustments, self.output_weights.T)
            h_adjustments2 = h_error2 * self.sigmoid_derivative(trained_hidden2)

            h_error = np.dot (h_error2 , self.hidden_weights2.T)
            h_adjustments1 = h_error * self.sigmoid_derivative(trained_hidden1)

            # Back propogate and adjust weights
            w1 = np.dot(self.inputs.T, h_adjustments1)
            w2 = np.dot(trained_hidden1.T, h_adjustments2)
            w3 = np.dot(trained_hidden2.T, o_adjustments)

            self.hidden_weights -= 0.0001 * w1
            self.hidden_weights2 -= 0.0001 * w2
            self.output_weights -= 0.0001 * w3
            print(i)

        return trained_outputs


    def sigmoid(self, x):
        a = 1 / (1 + np.exp (-x))
        return a


    def sigmoid_derivative(self, x):
        return x * (1 - x)