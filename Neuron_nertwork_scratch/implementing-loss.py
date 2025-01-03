import math
import numpy as np
import nnfs
from nnfs.datasets import spiral_data

nnfs.init()

class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.10 *np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))
    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases


class Activation_ReLu:
    def forward(self, inputs):
        self.output = np.maximum(0, inputs)

class Activation_Softmax:
    def forward(self, inputs):
        exp_values = np.exp(inputs - np.max(inputs, axis=1,keepdims=True))
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.output = probabilities

class Loss:
    def calculate(self, output, y):     # y = target output
        sample_losses = self.forward(output, y)
        data_loss = np.mean(sample_losses)
        return data_loss

class Loss_CategoricalCrossEntropy(Loss):
    def forward(self, y_pred, y_target):
        samples = len(y_pred)
#clipp these values to a number close to zero but but not equals to zero
#this is done to avoid zeros inputs, since they introduce problem of infinite
        y_pred_clipped = np.clip(y_pred, 1e-7, 1-1e-7)

# if y_target is in the scaler form
        if len(y_target.shape) == 1:
            correct_confidences = y_pred_clipped[range(samples),y_target]
        #if y_target is a vector
        elif len(y_target.shape) == 2:
            correct_confidences = np.sum(y_pred_clipped*y_target, axis=1)

        negative_log_likelihoods = -np.log(correct_confidences)
        return negative_log_likelihoods



X, y = spiral_data(samples=100, classes=3)

dense1 = Layer_Dense(2, 3)
activation1 = Activation_ReLu()

dense2 = Layer_Dense(3, 3)
activation2 = Activation_Softmax()

dense1.forward(X)
activation1.forward(dense1.output)

dense2.forward(activation1.output)
activation2.forward(dense2.output)

#print(activation2.output)

# to print the first five output of the batch
print(activation2.output[:5])

loss_function = Loss_CategoricalCrossEntropy()
loss = loss_function.calculate(activation2.output, y)

print("Loss:", loss)
