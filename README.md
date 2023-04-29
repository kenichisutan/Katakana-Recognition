# Katakana-Recognition

 A neural-network using numPy implementing backpropagation to recognize Japanese Katakana characters.

Network was originally made to be able to recognize 5 katakana characters (カキクケコ) rendered as a 5x5 array of binary inputs. However, it has since been expanded to cover all 46 characters that exist in the script. Previous versions remain in the repository under a designated folder.

The final network implements an input layer of 25 neurons, with a hidden layer of 25 neurons, and an output layer of 46 neurons.
![alt text](https://i.imgur.com/WAkAgq8.png)
![alt text](https://i.imgur.com/nHX44MD.png)

Two main files exist: 1 implementing NumPy and 1 without NumPy. Performance with NumPy increases dramatically and is recommended. However, both are funcitonal and similar in accuracy.
