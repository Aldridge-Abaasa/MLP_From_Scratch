#ifndef MLP_HPP
#define MLP_HPP

#include <vector>

// Activation functions
double sigmoid(double x);
double sigmoid_derivative(double x);
double tanh_derivative(double x);

// MLP class declaration
class MLP {
private:
    std::vector<std::vector<double>> weights_input_hidden;
    std::vector<std::vector<double>> weights_hidden_output;
    std::vector<double> hidden_layer;
    std::vector<double> output_layer;
    double learning_rate;

    void initialize_weights(std::vector<std::vector<double>>& weights, int rows, int cols);
    void forward(const std::vector<double>& input);
    double backpropagate(const std::vector<double>& input, const std::vector<double>& target);

public:
    MLP(int input_size, int hidden_size, int output_size, double learning_rate = 0.1);
    std::vector<double> predict(const std::vector<double>& input);
    void train(const std::vector<std::vector<double>>& inputs, const std::vector<std::vector<double>>& targets, int epochs);
};

#endif
