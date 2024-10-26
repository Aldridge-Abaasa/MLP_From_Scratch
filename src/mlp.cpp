#include "mlp.hpp"
#include <iostream>
#include <cmath>
#include <cstdlib>
#include <ctime>

// Activation functions
double sigmoid(double x) { return 1 / (1 + std::exp(-x)); }
double sigmoid_derivative(double x) { return x * (1 - x); }
double tanh_derivative(double x) { return 1 - x * x; }

// Constructor
MLP::MLP(int input_size, int hidden_size, int output_size, double learning_rate)
    : hidden_layer(hidden_size), output_layer(output_size), learning_rate(learning_rate) {
    initialize_weights(weights_input_hidden, input_size, hidden_size);
    initialize_weights(weights_hidden_output, hidden_size, output_size);
}

// Initialize weights
void MLP::initialize_weights(std::vector<std::vector<double>>& weights, int rows, int cols) {
    weights.resize(rows, std::vector<double>(cols));
    for (auto& row : weights) {
        for (auto& weight : row) {
            weight = ((double)rand() / RAND_MAX) * 0.2 - 0.1; // Small random values
        }
    }
}

// Forward propagation
void MLP::forward(const std::vector<double>& input) {
    for (size_t i = 0; i < hidden_layer.size(); ++i) {
        hidden_layer[i] = 0.0;
        for (size_t j = 0; j < input.size(); ++j) {
            hidden_layer[i] += input[j] * weights_input_hidden[j][i];
        }
        hidden_layer[i] = sigmoid(hidden_layer[i]);
    }

    for (size_t i = 0; i < output_layer.size(); ++i) {
        output_layer[i] = 0.0;
        for (size_t j = 0; j < hidden_layer.size(); ++j) {
            output_layer[i] += hidden_layer[j] * weights_hidden_output[j][i];
        }
        output_layer[i] = sigmoid(output_layer[i]);
    }
}

// Backpropagation
double MLP::backpropagate(const std::vector<double>& input, const std::vector<double>& target) {
    std::vector<double> output_errors(output_layer.size());
    for (size_t i = 0; i < output_layer.size(); ++i) {
        double error = target[i] - output_layer[i];
        output_errors[i] = error * sigmoid_derivative(output_layer[i]);
    }

    std::vector<double> hidden_errors(hidden_layer.size());
    for (size_t i = 0; i < hidden_layer.size(); ++i) {
        double error = 0.0;
        for (size_t j = 0; j < output_layer.size(); ++j) {
            error += output_errors[j] * weights_hidden_output[i][j];
        }
        hidden_errors[i] = error * sigmoid_derivative(hidden_layer[i]);
    }

    for (size_t i = 0; i < hidden_layer.size(); ++i) {
        for (size_t j = 0; j < output_layer.size(); ++j) {
            weights_hidden_output[i][j] += learning_rate * output_errors[j] * hidden_layer[i];
        }
    }

    for (size_t i = 0; i < input.size(); ++i) {
        for (size_t j = 0; j < hidden_layer.size(); ++j) {
            weights_input_hidden[i][j] += learning_rate * hidden_errors[j] * input[i];
        }
    }

    double total_error = 0.0;
    for (double error : output_errors) {
        total_error += std::abs(error);
    }
    return total_error;
}

// Predict
std::vector<double> MLP::predict(const std::vector<double>& input) {
    forward(input);
    return output_layer;
}

// Train
void MLP::train(const std::vector<std::vector<double>>& inputs, const std::vector<std::vector<double>>& targets, int epochs) {
    for (int epoch = 0; epoch < epochs; ++epoch) {
        double total_error = 0.0;
        for (size_t i = 0; i < inputs.size(); ++i) {
            forward(inputs[i]);
            total_error += backpropagate(inputs[i], targets[i]);
        }
        if (epoch % 100 == 0)
            std::cout << "Epoch " << epoch << " - Error: " << total_error << std::endl;
    }
}
