import numpy as np

inputs = [
            [1,2,3,2.5],
            [2.0, 5.0, -1.0, 2.0],
            [-1.5, 2.7, 3.3, -0.8]
        ]

#dot product would not work here, because of shape, thus we transpose weights
weights1 = [
            [0.2, 0.8, -0.5, 1.0],
            [0.5, -0.91, 0.26, -0.5],
            [-0.26, -0.27, 0.17, 0.87]
        ]
biases1 = [2, 3, 0.5]

weights2 = [
            [0.1, -0.14, 0.5],
            [-0.5, 0.12, -0.33],
            [-0.44, 0.73, -0.13]
        ]

biases2 = [-1, 2, -0.5]

output_layer1 = np.dot(inputs, np.array(weights1).T) + biases1
output_layer2 = np.dot(output_layer1, np.array(weights2).T) + biases2

print(output_layer2)
