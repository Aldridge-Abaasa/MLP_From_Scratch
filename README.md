# MLP_From_Scratch
Multi layered perceptron from scratch using C++

This project is a simple implementation of a Multi-Layer Perceptron (MLP) built from scratch in C++ without using any external machine learning libraries. It demonstrates the basics of neural networks, including forward propagation, backpropagation, and training on sample data.

## Features

- Create an MLP with a customizable number of inputs, hidden units, and outputs
- Initialize weights to small random values
- Predict output for a given input vector
- Train the MLP using backpropagation
- Example training on the XOR function and a custom function based on `sin(x1 - x2 + x3 - x4)`

## Project Structure

```plaintext
MLP_From_Scratch/
├── src/
│   └── mlp.cpp                 # Main C++ code for the MLP
├── include/
│   └── mlp.hpp                  # Header file with class declarations (optional if you split the code)
├── README.md                    # Project documentation
└── .gitignore                   # Git ignore file
