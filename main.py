import matplotlib.pyplot as plt
from functions import characters
from functions import functions as f
import numpy as np
import time


print("Training for all characters...")

# Create a list of characters and a list of their corresponding labels
xNames = ["A", "I", "U", "E", "O", "Ka", "Ki", "Ku", "Ke", "Ko",
          "Sa", "Shi", "Su", "Se", "So", "Ta", "Chi", "Tsu", "Te", "To",
          "Na", "Ni", "Nu", "Ne", "No", "Ha", "Hi", "Fu", "He", "Ho",
          "Ma", "Mi", "Mu", "Me", "Mo", "Ya", "Yu", "Yo",
          "Ra", "Ri", "Ru", "Re", "Ro", "Wa", "Wo", "N"]
x = [ characters.a(), characters.i(), characters.u(),
      characters.e(), characters.o(), characters.ka(),
      characters.ki(), characters.ku(), characters.ke(),
      characters.ko(), characters.sa(), characters.shi(),
      characters.su(), characters.se(), characters.so(),
      characters.ta(), characters.chi(), characters.tsu(),
      characters.te(), characters.to(), characters.na(),
      characters.ni(), characters.nu(), characters.ne(),
      characters.no(), characters.ha(), characters.hi(),
      characters.fu(), characters.he(), characters.ho(),
      characters.ma(), characters.mi(), characters.mu(),
      characters.me(), characters.mo(), characters.ya(),
      characters.yu(), characters.yo(), characters.ra(),
      characters.ri(), characters.ru(), characters.re(),
      characters.ro(), characters.wa(), characters.wo(),
      characters.n()]

targets = []

for i in range(len(x)):
    targets.append([])
    targets[i] = [0] * len(x)
    targets[i][i] = 1

print("Input:")
for i in range(len(x)):
    print(xNames[i] + ":", x[i])
print()

# Convert the lists to numpy arrays
x = np.array(x)
targets = np.array(targets)

# Define the variables
inputLayerNeurons, hiddenLayerNeurons, outputLayerNeurons = 25, 25, 46
learningRate = 0.5
iterations = 3000

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
startTime = time.time()
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
endTime = time.time()
print("Training time:", endTime - startTime, "seconds")
print()

# Error
print("Error:")
for i in range(len(x)):
    print(xNames[i] + ":", MSE[i][iterations - 1])
print()

# Testing
test = [characters.ki(), characters.ma(), characters.fu()]
testNames = ["Ki", "Ma", "Fu"]
testTargets = [[0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]

# Print the targets
print("Targets:")
for i in range(len(test)):
    print(testNames[i] + ":", testTargets[i])
print()

# Print the results
for i in range(len(test)):
    hiddenLayers = f.sigmoidEstimation(np.dot(test[i], weightsInputHidden) + biasHidden)
    output = f.sigmoidEstimation(np.dot(hiddenLayers, weightsHiddenOutput) + biasOutput)
    print("Result " + testNames[i] + ":\n", output)

# Plotting
for i in range(len(MSE)):
    plt.plot(MSE[i], label=xNames[i])

# Labels
title = "Learning rate: " + str(learningRate) + ", Iterations: " + str(iterations)
plt.title(title)
plt.xlabel("Iterations")
plt.ylabel("Mean Squared Error")

# Legend
#plt.legend()
plt.show()