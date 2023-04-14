import matplotlib.pyplot as plt
from functions import characters
from functions import functions as f
import numpy as np


print("Training for all characters...")

# Create a list of characters and a list of their corresponding labels
xNames = ["Ka", "Ki", "Ku", "Ke", "Ko", "Sa", "Shi", "Su", "Se", "So"]
x = [characters.ka(), characters.ki(), characters.ku(),
     characters.ke(), characters.ko(), characters.sa(),
     characters.shi(), characters.su(), characters.se(),
     characters.so()]
targets = [[1, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 1, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 1]]

print("Input:")
for i in range(len(x)):
    print(xNames[i] + ":", x[i])
print()

# Convert the lists to numpy arrays
x = np.array(x)
targets = np.array(targets)

# Define the variables
inputLayerNeurons, hiddenLayerNeurons, outputLayerNeurons = 25, 20, 10
learningRate = 0.5
iterations = 5000

# MSE per input per iteration
MSE = []
for i in range(len(x)):
    MSE.append([])
    MSE[i] = [0] * iterations

# Weights
# Input to hidden layer
weightsInputHidden = np.random.uniform(0, 1, size=(inputLayerNeurons, hiddenLayerNeurons))
# Hidden layer to output
weightsHiddenOutput = np.random.uniform(0, 1, size=(hiddenLayerNeurons, outputLayerNeurons))

# Define bias (randomly initialized)
# Hidden layer
biasHidden = np.random.uniform(0, 1, size=hiddenLayerNeurons)
# Output layer
biasOutput = np.random.uniform(0, 1, size=outputLayerNeurons)

# Training loop
for i in range(iterations):
    # For each input
    for j in range(len(targets)):
        # Forward propagation
        hiddenLayers = f.sigmoidEstimation(np.dot(x[j], weightsInputHidden) + biasHidden)
        # Observed output
        observed = f.sigmoidEstimation(np.dot(hiddenLayers, weightsHiddenOutput) + biasOutput)
        # Error estimation
        error = targets[j] - observed
        # Mean Squared Error (per input per iteration)
        MSE[j][i] = (np.sum(error ** 2)) / len(error)
        # Back propagation
        dOutputError = error * f.sigmoidDerivative(observed)
        dHiddenError = np.dot(dOutputError, weightsHiddenOutput.T) * f.sigmoidDerivative(hiddenLayers)
        # Updating weights and biases
        weightsHiddenOutput += np.dot(hiddenLayers.reshape(hiddenLayerNeurons, 1), dOutputError.reshape(1, outputLayerNeurons)) * learningRate
        biasOutput += dOutputError * learningRate
        weightsInputHidden += np.dot(x[j].reshape(inputLayerNeurons, 1), dHiddenError.reshape(1, hiddenLayerNeurons)) * learningRate
        biasHidden += dHiddenError * learningRate

# Testing
test = [characters.ka(), characters.ki(), characters.ku(), characters.ke(), characters.ko(),
        characters.sa(), characters.shi(), characters.su(), characters.se(), characters.so()]

# Print the targets
print("Targets:")
for i in range(len(test)):
    print(xNames[i] + ":", targets[i])

# Print the results
for i in range(len(test)):
    hiddenLayers = f.sigmoidEstimation(np.dot(test[i], weightsInputHidden) + biasHidden)
    output = f.sigmoidEstimation(np.dot(hiddenLayers, weightsHiddenOutput) + biasOutput)
    print("Result " + xNames[i] + ":\n", output)

# Plotting
for i in range(len(MSE)):
    plt.plot(MSE[i], label=xNames[i])

# Labels
title = "Learning rate: " + str(learningRate) + ", Iterations: " + str(iterations)
plt.title(title)
plt.xlabel("Iterations")
plt.ylabel("Mean Squared Error")

# Legend
plt.legend()
plt.show()