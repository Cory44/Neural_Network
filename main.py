import numpy as np
import pickle
from NeuralNet import NeuralNet

#########################################
# Load in the image arrays for the
# training and test data from mnist.pkl
#########################################
def load():
    with open("mnist.pkl", 'rb') as f:
        mnist = pickle.load(f)
    return mnist["training_images"], mnist["training_labels"], mnist["test_images"], mnist["test_labels"]


##############################################
# Import and format data, train network
##############################################

# Load the training and test data
x_train, t_train, x_test, t_test = load()

# Format input and output data for training
inputs = x_train/256

outputs = np.zeros((60000, 10), dtype=int)
for i in range(60000):
    x = t_train[int(i)]
    outputs[i][x] = 1

# Train network
neural_net = NeuralNet(2, inputs, outputs)
trained_set = neural_net.train(100)
trained_set = np.around(trained_set, decimals=2)


##############################################
# Calculate training accuracy and print
# final iteration training results
##############################################
accuracy = 0
for i in range(60000):
    trained = np.rint(trained_set[i]).astype(int)
    actual = outputs[i]
    same = False
    if np.array_equal(trained, actual):
        accuracy += 1
        same = True
    # print(i, trained, actual, same)
print(accuracy/60000)
