from NeuralNet import NeuralNet
import numpy as np
import pickle
from matplotlib import pyplot as plt

def load():
    with open("mnist.pkl", 'rb') as f:
        mnist = pickle.load(f)
    return mnist["training_images"], mnist["training_labels"], mnist["test_images"], mnist["test_labels"]

x_train, y_train, x_test, y_test = load()

outputs = np.zeros((60000, 10), dtype=int)
for i in range(60000):
    x = y_train[int(i)]
    outputs[i][x] = 1

net = NeuralNet(x_train, outputs)

first_image = x_test[9998]
first_image = np.array(first_image, dtype='uint8')
pixels = first_image.reshape((28, 28))
plt.imshow(pixels, cmap='gray')
plt.show()

# for i in range(28):
#     for j in range(28):
#
#         if pixels[i][j] < 128:
#             print(" ", end="\t")
#         else:
#             print("@", end="\t")
#     print()

trained = net.train(100)


images = net.bulk_classify(x_test)

properly_classified = 0

for i in range(len(y_test)):
    print(images[i], y_test[i], np.argmax(images[i]) == y_test[i])
    if np.argmax(images[i]) == y_test[i]:
        properly_classified += 1

print(properly_classified / len(y_test))