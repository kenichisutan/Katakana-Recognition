import matplotlib.pyplot as plt
from functions import characters
import random
import time
from functions import functions as f

start_time = time.time()
print("Start time;", start_time)
print("Training for all characters...")

xNames = [ "A", "I", "U", "E", "O", "Ka", "Ki", "Ku", "Ke", "Ko",
           "Sa", "Shi", "Su", "Se", "So", "Ta", "Chi", "Tsu", "Te", "To",
           "Na", "Ni", "Nu", "Ne", "No", "Ha", "Hi", "Fu", "He", "Ho",
           "Ma", "Mi", "Mu", "Me", "Mo", "Ya", "Yu", "Yo",
           "Ra", "Ri", "Ru", "Re", "Ro", "Wa", "Wo", "N" ]
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
      characters.n() ]

targets = [ ]
for i in range(len(x)):
    targets.append([ ])
    targets[ i ] = [ 0 ] * 46
    targets[ i ][ i ] = 1

print("Input:")
for i in range(len(x)):
    print(xNames[ i ] + ":", x[ i ])
print()

# Define the variables
inputLayerNeurons, hiddenLayerNeurons, outputLayerNeurons = 25, 25, 46
learningRate = 0.5
iterations = 2500

# MSE per input per iteration
MSE = [ ]
for i in range(len(x)):
    MSE.append([ ])
    MSE[ i ] = [ 0 ] * iterations

# Define weights (randomly initialized)
# Input to hidden layer
weightsInputHidden = [ ]
for j in range(inputLayerNeurons):
    row = [ ]
    for i in range(hiddenLayerNeurons):
        row.append(random.uniform(0, 1))
    weightsInputHidden.append(row)
# Hidden layer to output
weightsHiddenOutput = [ ]
for j in range(hiddenLayerNeurons):
    row = [ ]
    for i in range(outputLayerNeurons):
        row.append(random.uniform(0, 1))
    weightsHiddenOutput.append(row)

# Define bias (randomly initialized)
# Hidden layer
biasHidden = [ ]
for i in range(hiddenLayerNeurons):
    biasHidden.append(random.uniform(0, 1))
# Output layer
biasOutput = [ ]
for i in range(outputLayerNeurons):
    biasOutput.append(random.uniform(0, 1))

# Training loop
for i in range(iterations):
    # For each input
    for j in range(len(targets)):

        # Forward propagation
        hiddenLayers = [ ]
        for l in range(hiddenLayerNeurons):
            sum_val = 0
            for k in range(inputLayerNeurons):
                sum_val += x[ j ][ k ] * weightsInputHidden[ k ][ l ]
            hidden_layer_val = f.sigmoidEstimation(sum_val + biasHidden[ l ])
            hiddenLayers.append(hidden_layer_val)

        # Observed output
        observed = [ ]
        for l in range(outputLayerNeurons):
            sum_val = 0
            for k in range(hiddenLayerNeurons):
                sum_val += hiddenLayers[ k ] * weightsHiddenOutput[ k ][ l ]
            output_val = f.sigmoidEstimation(sum_val + biasOutput[ l ])
            observed.append(output_val)

        # Error estimation
        error = [ ]
        for k in range(outputLayerNeurons):
            error_val = targets[ j ][ k ] - observed[ k ]
            error.append(error_val)

        # Mean Squared Error (per input per iteration)
        MSE[ j ][ i ] = 0
        for k in range(outputLayerNeurons):
            MSE[ j ][ i ] += error[ k ] ** 2
        MSE[ j ][ i ] /= len(error)

        # Back propagation
        dOutputError = [ ]
        for k in range(outputLayerNeurons):
            d_output_error_val = error[ k ] * f.sigmoidDerivative(observed[ k ])
            dOutputError.append(d_output_error_val)
        dHiddenError = [ ]
        for k in range(hiddenLayerNeurons):
            sum_val = 0
            for l in range(outputLayerNeurons):
                sum_val += dOutputError[ l ] * weightsHiddenOutput[ k ][ l ]
            d_hidden_error_val = sum_val * f.sigmoidDerivative(hiddenLayers[ k ])
            dHiddenError.append(d_hidden_error_val)

        # Updating weights and biases
        for k in range(hiddenLayerNeurons):
            for l in range(outputLayerNeurons):
                weightsHiddenOutput[ k ][ l ] += hiddenLayers[ k ] * dOutputError[ l ] * learningRate
        for l in range(outputLayerNeurons):
            biasOutput[ l ] += dOutputError[ l ] * learningRate
        for k in range(inputLayerNeurons):
            for l in range(hiddenLayerNeurons):
                weightsInputHidden[ k ][ l ] += x[ j ][ k ] * dHiddenError[ l ] * learningRate
        for l in range(hiddenLayerNeurons):
            biasHidden[ l ] += dHiddenError[ l ] * learningRate

# Testing
test = [ characters.ki(), characters.ma(), characters.fu() ]
testNames = [ "Ki", "Ma", "Fu" ]
testTargets = [ [ 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                  0, 0, 0, 0, 0, 0, 0, 0, 0, 0 ],
                [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0,
                  0, 0, 0, 0, 0, 0, 0, 0, 0, 0 ],
                [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                  0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
                  0, 0, 0, 0, 0, 0, 0, 0, 0, 0 ] ]

# Print the targets
print("Targets:")
for i in range(len(test)):
    print(xNames[ i ] + ":", targets[ i ])

for i in range(len(test)):
    # Compute the hidden layer
    hiddenLayer = [ ]
    for j in range(hiddenLayerNeurons):
        neuronOutput = 0
        for k in range(inputLayerNeurons):
            neuronOutput += test[ i ][ k ] * weightsInputHidden[ k ][ j ]
        neuronOutput += biasHidden[ j ]
        hiddenLayer.append(f.sigmoidEstimation(neuronOutput))

    # Compute the output layer
    outputLayer = [ ]
    for j in range(outputLayerNeurons):
        neuronOutput = 0
        for k in range(hiddenLayerNeurons):
            neuronOutput += hiddenLayer[ k ] * weightsHiddenOutput[ k ][ j ]
        neuronOutput += biasOutput[ j ]
        outputLayer.append(f.sigmoidEstimation(neuronOutput))
    print("Result " + testNames[ i ] + ":\n", observed)

# Plotting
for i in range(len(MSE)):
    plt.plot(MSE[ i ], label=xNames[ i ])

print("Program ran in %s seconds" % (time.time() - start_time))

# Labels
title = "Learning rate: " + str(learningRate) + ", Iterations: " + str(iterations)
plt.title(title)
plt.xlabel("Iterations")
plt.ylabel("Mean Squared Error")

# Legend
# plt.legend()
plt.show()