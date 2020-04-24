# Neural Network
###### A from scratch Convolutional Neural Network class

The NerualNet class has been developed to be as flexible as possible. A user can instantiate the network with numpy 
arrays of varying sizes for both the inputs and outputs.
I have given an example using the MNIST database of handwritten digits, which consists of a set of 60,000 images, 
each 784 pixels (28 x 28), and the corresponding outputs which are numpy arrays with a length of 10 (with the index 
representing the correct digit from 0 to 9).

With the MNIST database of handwritten data as well as tests with the MNIST Fashion database (consisting of 28 X 28 
pixel images of clothing items), I have gotten the network to classify new items with an accuracy >95% at 10,000 
training iterations.


#### Specs of the class
- Uses numpy arrays for efficient vector calculations
- Trains using user given inputs and outputs for a specified number of iterations
- Instance variables for the node weights are created and are updated at each training iteration
- Classification can be done for an individual item or in bulk using a user given numpy array of values to classify 

#### Special Thanks
- [Yann LeCun](http://yann.lecun.com), Corinna Cortes and Christopher Burges and their [MNIST database of handwritten digits](http://yann.lecun.com/exdb/mnist/)
- [hsjeong5](https://github.com/hsjeong5) and their [MNIST-for-Numpy](https://github.com/hsjeong5/MNIST-for-Numpy) project for helping me download the MNIST database and loading it into my Neural Network
- A huge shout out to [3Blue1Brown](https://www.youtube.com/channel/UCYO_jab_esuFRV4b17AJtAw) and thier amazing Youtube series on [Neural Networks](https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi)
- And finally [Addy](https://stackoverflow.com/users/3399825/addy) on Stackoverflow for helping me solve [this](https://stackoverflow.com/questions/56740145/neural-network-issue-with-back-propagation-calculation) nagging back propagation error
