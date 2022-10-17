import numpy as np


class NeuralNetwork:

    def __init__(self, layer_sizes):
        """
        Neural Network initialization.
        Given layer_sizes as an input, you have to design a Fully Connected Neural Network architecture here.
        :param layer_sizes: A list containing neuron numbers in each layers. For example [3, 10, 2] means that there are
        3 neurons in the input layer, 10 neurons in the hidden layer, and 2 neurons in the output layer.
        """
        # TODO (Implement FCNNs architecture here)
        self.W = []
        self.b = []
        mu, sigma = 0, 1
        layer_num = len(layer_sizes)

        for i in range(layer_num - 1):
            j = layer_sizes[i+1]
            k = layer_sizes[i]
            self.W.append(np.random.normal(mu, sigma, (j,k)))
            self.b.append(np.zeros((layer_sizes[i + 1], 1)))

        # self.activation = np.vectorize(self.activation)
        # pass

    def activation(self, x):
        """
        The activation function of our neural network, e.g., Sigmoid, ReLU.
        :param x: Vector of a layer in our network.
        :return: Vector after applying activation function.
        """
        # TODO (Implement activation function here)
        # Sigmoid (needs to be checked later)
        return 1 / (1 + np.exp(-x))
        # pass

    def forward(self, x):
        """
        Receives input vector as a parameter and calculates the output vector based on weights and biases.
        :param x: Input vector which is a numpy array.
        :return: Output vector
        """
        # TODO (Implement forward function here)
        output = x
        num_of_w = len(self.W)
        # calculate output pf each layer
        for i in range(num_of_w):
            z = (self.W[i] @ output) + self.b[i]
            output = self.activation(z)
        return output
        # pass
