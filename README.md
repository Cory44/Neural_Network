# Neural Network
###### A from scratch Neural Network using the MNIST handwritten digit datatbase for training

This is a simple Neural Network, built completely by me. The goal of the Neural Network is to identify hand written digits. I have used the MNIST handwritten digit database to train the network.

The network will train based on the user specifed number of training iterations. The resulting weights for both the hidden layer and output nodes will be saved in a .txt file for use later. 

At 25,000 iterations the network correctly identified >94% of the 60,000 digit images in the training set and had >91% accuracy when using the weightings to identify the 10,000 test images (**NOTE:** 25,000 training iterations took ~2 hours to run on my laptop :grimacing:!)

#### Specs of the newtork
- 784 input nodes (28 x 28 pixel images of digits)
- 16 node hidden layer
- 10 node output layer

#### Special Thanks
- [Yann LeCun](http://yann.lecun.com), Corinna Cortes and Christopher Burges and their [MNIST database of handwritten digits](http://yann.lecun.com/exdb/mnist/)
- [hsjeong5](https://github.com/hsjeong5) and their [MNIST-for-Numpy](https://github.com/hsjeong5/MNIST-for-Numpy) project for helping me download the MNIST database and loading it into my Neural Network
- A huge shout out to [3Blue1Brown](https://www.youtube.com/channel/UCYO_jab_esuFRV4b17AJtAw) and thier amazing Youtube series on [Neural Networks](https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi)
- And finally [Addy](https://stackoverflow.com/users/3399825/addy) on Stackoverflow for helping me solve [this](https://stackoverflow.com/questions/56740145/neural-network-issue-with-back-propagation-calculation) nagging back propagation error
