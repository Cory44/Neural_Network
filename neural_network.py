import numpy as np
import pickle


#########################################
# Normalization (Sigmoid) and
# derivative functions
#########################################
def sigmoid(x):
    a = 1 / (1 + np.exp(-x))
    return a

def sigmoid_derivative(x):
    return x * (1 - x)


#########################################
# Load in the image arrays for the
# training and test data from mnist.pkl
#########################################
def load():
    with open("mnist.pkl",'rb') as f:
        mnist = pickle.load(f)
    return mnist["training_images"], mnist["training_labels"], mnist["test_images"], mnist["test_labels"]


#########################################
# Training function
#########################################
def train(n, inputs, outputs):
    input_layer = inputs
    trained_outputs = outputs

    # Create weights for hidden and output layers
    weights1 = 2 * np.random.random ((784 , 16)) - 1
    weights2 = 2 * np.random.random ((16 , 10)) - 1

    # Loop through traingin iterations
    for i in range(n):
        # Times input by weights, then normalize using sigmoid
        # on both hidden and output layers
        trained_hidden = sigmoid(np.dot(input_layer, weights1))
        trained_outputs = sigmoid(np.dot(trained_hidden, weights2))

        # Caluclate error and adjusment on output layer
        o_error = trained_outputs - outputs
        o_adjustments = o_error * sigmoid_derivative(trained_outputs)

        # Calculate error and sdjustment on hidden layer
        h_error = np.dot(o_adjustments, weights2.T)
        h_adjustments = h_error * sigmoid_derivative(trained_hidden)

        # Back propogate and adjust weights
        w1 = np.dot(input_layer.T, h_adjustments)
        w2 = np.dot (trained_hidden.T , o_adjustments)
        weights1 -= 0.0001 * w1
        weights2 -= 0.0001 * w2
        print("1")

    return trained_outputs, weights1, weights2


##############################################
# Import and format data, train network
##############################################

# Load the training and test data
x_train, t_train, x_test, t_test = load()

# Format input and output data for training
inputs = x_train/256

outputs = np.zeros((60000,10), dtype=int)
for i in range(60000):
    x = t_train[int(i)]
    outputs[i][x] = 1

# Train network
trained_set, weights1, weights2 = train(10000, inputs, outputs)
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
    print(i, trained, actual, same)
print(accuracy/60000)


##############################################
# Write weights to a text file, these can be
# used to test data later
##############################################
w = open("weights.txt", "w+")
for i in weights1:
    w.write(np.array2string(i))
w.write("\n")
w.write(np.array2string(weights2) + "\n")


#############################################
# Test network weights using MNIST test set
#############################################
'''
inputs_test = x_test/256
outputs_test = np.zeros((10000,10), dtype=int)

for i in range(10000):
    x = t_test[int(i)]
    outputs_test[i][x] = 1

trained_hidden = sigmoid (np.dot(inputs_test, weights1))
trained_outputs = np.around(sigmoid (np.dot(trained_hidden , weights2)), decimals=2)

test_accuracy = 0
for i in range(10000):
    same = False
    if np.array_equal(np.rint(trained_outputs[i]).astype(int), outputs_test[i]):
        test_accuracy += 1
        same = True
    print(i, np.rint(trained_outputs[i]).astype(int), outputs_test[i], same)

print(test_accuracy/10000)
'''