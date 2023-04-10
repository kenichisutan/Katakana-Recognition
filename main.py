import matplotlib.pyplot as plt
import characters
import numpy as np
import functions as f

print("Testing for Ka character")
print("Input:")
print(characters.ka())
print(characters.ki())
print(characters.ku())
print(characters.ke())
print(characters.ko())

# Create a list of characters and a list of their corresponding labels
x = [characters.ka(), characters.ki(), characters.ku(),
     characters.ke(), characters.ko()]
targets = [[[1], [0], [0], [0], [0]], [[0], [1], [0], [0], [0]], [[0], [0], [1], [0], [0]],
           [[0], [0], [0], [1], [0]], [[0], [0], [0], [0], [1]]]

# Convert the lists to numpy arrays
x = np.array(x)
targets = np.array(targets)

# Define the variables
inputLayerNeurons, hiddenLayerNeurons, outputLayerNeurons = 25, 5, 5
learningRate = 1
iterations = 5000
MSE = [0] * iterations

# Create a list of weights and biases
outWeights = np.random.uniform(size=(hiddenLayerNeurons, outputLayerNeurons))
outBias = np.random.uniform(size=(1, outputLayerNeurons))

# Create a list of weights and biases for the hidden layer
hiddenWeights = np.random.uniform(size=(inputLayerNeurons, hiddenLayerNeurons))
hiddenBias = np.random.uniform(size=(1, hiddenLayerNeurons))

for i in range(iterations):
    observedOutput = [0, 0, 0, 0, 0]
    for j in range(len(targets)):
        # Forward Propagation
        hidLayerActivation = np.dot(x, hiddenWeights)
        hidLayerActivation += hiddenBias
        hidLayerOutput = f.sigmoidEstimation(hidLayerActivation)

        outLayerActivation = np.dot(hidLayerOutput, outWeights[j])
        outLayerActivation = [outLayerActivation]
        outLayerActivation += outBias
        observedOutput[j] = f.sigmoidEstimation(outLayerActivation)

        # Backpropagation
        error = targets[j] - observedOutput[j]
        MSE[i] = (np.sum(error ** 2)) / len(error)
        dObservedOutput = error * f.sigmoidDerivative(observedOutput[j])

        hidLayerError = dObservedOutput.dot(outWeights[j].T)
        dHidLayer = hidLayerError * f.sigmoidDerivative(hidLayerOutput)

        # Updating Weights and Biases
        outWeights[j] += hidLayerOutput.dot(dObservedOutput) * learningRate
        outBias[j] += np.sum(dObservedOutput, axis=0, keepdims=True) * learningRate
        hiddenWeights += x.T.dot(dHidLayer) * learningRate
        hiddenBias += np.sum(dHidLayer, axis=0, keepdims=True) * learningRate

print("Output weights:", outWeights)

print("Target:", targets)
print("Output:", observedOutput)

# Plotting
plt.plot(MSE)
plt.xlabel("Iterations")
plt.ylabel("Mean Squared Error")
plt.show()