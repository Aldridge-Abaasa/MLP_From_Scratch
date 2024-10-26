#include "include/mlp.hpp"
#include <iostream>

int main() {
    srand(time(0));  // Seed for random weight initialization

    // XOR training data
    std::vector<std::vector<double>> inputs = { {0, 0}, {0, 1}, {1, 0}, {1, 1} };
    std::vector<std::vector<double>> targets = { {0}, {1}, {1}, {0} };

    // Initialize the MLP
    MLP mlp(2, 3, 1, 0.1);  // 2 inputs, 3 hidden units, 1 output, learning rate of 0.1

    // Train the MLP
    mlp.train(inputs, targets, 1000);

    // Test the MLP
    std::cout << "Predictions:" << std::endl;
    for (const auto& input : inputs) {
        std::vector<double> prediction = mlp.predict(input);
        std::cout << "(" << input[0] << ", " << input[1] << ") -> " << prediction[0] << std::endl;
    }

    return 0;
}
