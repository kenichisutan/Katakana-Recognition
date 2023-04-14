import matplotlib.pyplot as plt
from functions import characters
import numpy as np
import functions as f

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
print(xNames[0] + ":", x[0])
print(xNames[1] + ":", x[1])
print(xNames[2] + ":", x[2])
print(xNames[3] + ":", x[3])
print(xNames[4] + ":", x[4])
print(xNames[5] + ":", x[5])
print(xNames[6] + ":", x[6])
print(xNames[7] + ":", x[7])
print(xNames[8] + ":", x[8])
print(xNames[9] + ":", x[9])
print()

# Convert the lists to numpy arrays
x = np.array(x)
targets = np.array(targets)

# Define the variables
inputLayerNeurons, hiddenLayerNeurons, outputLayerNeurons = 25, 20, 10
learningRate = 1
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
print("Targets:")
print("Ka:", targets[0])
print("Ki:", targets[1])
print("Ku:", targets[2])
print("Ke:", targets[3])
print("Ko:", targets[4])
print("Sa:", targets[5])
print("Shi:", targets[6])
print("Su:", targets[7])
print("Se:", targets[8])
print("So:", targets[9])
test = [characters.ka(), characters.ki(), characters.ku(), characters.ke(), characters.ko(),
        characters.sa(), characters.shi(), characters.su(), characters.se(), characters.so()]

hiddenLayers = f.sigmoidEstimation(np.dot(test[0], weightsInputHidden) + biasHidden)
outputKa = f.sigmoidEstimation(np.dot(hiddenLayers, weightsHiddenOutput) + biasOutput)

print("Result Ka:\n", outputKa)

hiddenLayers = f.sigmoidEstimation(np.dot(test[1], weightsInputHidden) + biasHidden)
outputKi = f.sigmoidEstimation(np.dot(hiddenLayers, weightsHiddenOutput) + biasOutput)

print("Result Ki:\n", outputKi)

hiddenLayers = f.sigmoidEstimation(np.dot(test[2], weightsInputHidden) + biasHidden)
outputKu = f.sigmoidEstimation(np.dot(hiddenLayers, weightsHiddenOutput) + biasOutput)

print("Result Ku:\n", outputKu)

hiddenLayers = f.sigmoidEstimation(np.dot(test[3], weightsInputHidden) + biasHidden)
outputKe = f.sigmoidEstimation(np.dot(hiddenLayers, weightsHiddenOutput) + biasOutput)

print("Result Ke:\n", outputKe)

hiddenLayers = f.sigmoidEstimation(np.dot(test[4], weightsInputHidden) + biasHidden)
outputKo = f.sigmoidEstimation(np.dot(hiddenLayers, weightsHiddenOutput) + biasOutput)

print("Result Ko:\n", outputKo)

hiddenLayers = f.sigmoidEstimation(np.dot(test[5], weightsInputHidden) + biasHidden)
outputSa = f.sigmoidEstimation(np.dot(hiddenLayers, weightsHiddenOutput) + biasOutput)

print("Result Sa:\n", outputSa)

hiddenLayers = f.sigmoidEstimation(np.dot(test[6], weightsInputHidden) + biasHidden)
outputShi = f.sigmoidEstimation(np.dot(hiddenLayers, weightsHiddenOutput) + biasOutput)

print("Result Shi:\n", outputShi)

hiddenLayers = f.sigmoidEstimation(np.dot(test[7], weightsInputHidden) + biasHidden)
outputSu = f.sigmoidEstimation(np.dot(hiddenLayers, weightsHiddenOutput) + biasOutput)

print("Result Su:\n", outputSu)

hiddenLayers = f.sigmoidEstimation(np.dot(test[8], weightsInputHidden) + biasHidden)
outputSe = f.sigmoidEstimation(np.dot(hiddenLayers, weightsHiddenOutput) + biasOutput)

print("Result Se:\n", outputSe)

hiddenLayers = f.sigmoidEstimation(np.dot(test[9], weightsInputHidden) + biasHidden)
outputSo = f.sigmoidEstimation(np.dot(hiddenLayers, weightsHiddenOutput) + biasOutput)

print("Result So:\n", outputSo)


# Plotting
plt.plot(MSE[0], label="Ka")
plt.plot(MSE[1], label="Ki")
plt.plot(MSE[2], label="Ku")
plt.plot(MSE[3], label="Ke")
plt.plot(MSE[4], label="Ko")
plt.plot(MSE[5], label="Sa")
plt.plot(MSE[6], label="Shi")
plt.plot(MSE[7], label="Su")
plt.plot(MSE[8], label="Se")
plt.plot(MSE[9], label="So")

# Labels
title = "Learning rate: " + str(learningRate) + ", Iterations: " + str(iterations)
plt.title(title)
plt.xlabel("Iterations")
plt.ylabel("Mean Squared Error")

# Legend
plt.legend()
plt.show()