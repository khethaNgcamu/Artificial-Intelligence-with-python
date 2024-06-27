#inputs = [1, 2, 3, 2.5]

#weight1 = [0.2, 0.8, -0.5, 1.0]
#weight2 = [0.5, -0.91, 0.26, -0.5]
#weight3 = [-0.26, -0.27, 0.17, 0.87]

#bias1 = 2
#bias2 = 3
#bias3 = 0.5

#outputs = [
 #          inputs[0] * weight1[0] + inputs[1] * weight1[1] + inputs[2] * weight1[2] + inputs[3] * weight1[3] + bias1,
  #         inputs[0] * weight2[0] + inputs[1] * weight2[1] + inputs[2] * weight2[2] + inputs[3] * weight2[3] + bias2,
   #        inputs[0] * weight3[0] + inputs[1] * weight3[1] + inputs[2] * weight3[2] + inputs[3] * weight3[3] + bias3
    #      ]

#print(outputs)



#efficient method:

inputs = [1, 2, 3, 2.5]

weights = [
     [0.2, 0.8, -0.5, 1.0],
     [0.5, -0.91, 0.26, -0.5],
     [-0.26, -0.27, 0.17, 0.87]
]

biases = [2, 3, 0.5]


layer_outputs = []
for neuron_weights, neuron_bias in zip(weights, biases):
    neuron_output = 0
    for n_input, weight in zip(inputs, neuron_weights):
        neuron_output += n_input * weight
    neuron_output += neuron_bias
    layer_outputs.append(neuron_output)

print(layer_outputs)
