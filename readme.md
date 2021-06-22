# CS50’s Introduction to Artificial Intelligence with Python
# Project 5: Neural Networks: Traffic Signs

**Aim**: Write an AI to identify which traffic sign appears in a photograph.

**Description**: As research continues in the development of self-driving cars, one of the key challenges is computer vision, allowing these cars to develop an understanding of their environment from digital images. One case is the ability to recognize and distinguish road signs. For this project, a TensorFlow neural network is built using Keras Sequential Model and then optimized for the task at hand, i.e. recognition of road signs from images.

First, a 2D convolutional layer is used (Conv2D) with 32 filters, a kernel size of 5 pixels, 2 strides and ReLU activation, followed by a MaxPooling2D with a pool size of 3 pixels. Then we add a flatten layer for input to the NN. Finally we add a Dense layer as output of the NN with softmax activation and the NUM_CATEGORIES nodes. We compile this NN using ADAM optimizer CategoricalCrossentropy losses and metrics for accuracy. The result is an accuracy of 75% on test data. Not bad, but could be improved.

Adding a hidden layer of 128 nodes, we manage to improve the accuracy on test data to 82%. Then, I decided to reduce the kernel size of the convolution layer from 5 to 3 pixels and remove the 2 strides to have the default behaviour. The accuracy on test data improved to 92%. Including a dropout layer after the convolution and max pool to avoid overfitting reached an accuracy of 94.5% on test data, which is already a great result.

Finally, adding a second convolution layer of kernel size 4 pixels and 32 filters after the first convolution and increasing the Epochs to 20 results in an almost perfect test data accuracy, with 98% accuracy!

More info on this problem set here: https://cs50.harvard.edu/ai/2020/projects/5/traffic/

**Cloning instructions**:
1) Clone this repository.
2) Download the <a href="https://cdn.cs50.net/ai/2020/x/projects/5/gtsrb.zip">data set</a> and extract it to the cloned repository folder.
3) Ready!

Usage:
```
python shopping.py data [model] [--options]  
```

Example:
```
$ python traffic.py gtsrb
Epoch 1/10
500/500 [==============================] - 5s 9ms/step - loss: 3.7139 - accuracy: 0.1545
Epoch 2/10
500/500 [==============================] - 6s 11ms/step - loss: 2.0086 - accuracy: 0.4082
Epoch 3/10
500/500 [==============================] - 6s 12ms/step - loss: 1.3055 - accuracy: 0.5917
Epoch 4/10
500/500 [==============================] - 5s 11ms/step - loss: 0.9181 - accuracy: 0.7171
Epoch 5/10
500/500 [==============================] - 7s 13ms/step - loss: 0.6560 - accuracy: 0.7974
Epoch 6/10
500/500 [==============================] - 9s 18ms/step - loss: 0.5078 - accuracy: 0.8470
Epoch 7/10
500/500 [==============================] - 9s 18ms/step - loss: 0.4216 - accuracy: 0.8754
Epoch 8/10
500/500 [==============================] - 10s 20ms/step - loss: 0.3526 - accuracy: 0.8946
Epoch 9/10
500/500 [==============================] - 10s 21ms/step - loss: 0.3016 - accuracy: 0.9086
Epoch 10/10
500/500 [==============================] - 10s 20ms/step - loss: 0.2497 - accuracy: 0.9256
333/333 - 5s - loss: 0.1616 - accuracy: 0.9535
```
