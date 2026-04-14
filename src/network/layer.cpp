#include "layer.hpp"
#include "../math/linAlg.hpp"

DenseLayer::DenseLayer(int inputs, int neurons) : biases(neurons) {
	weights = 0.01 * randomMatrix(inputs, neurons);
}

void DenseLayer::forward(const Matrix& inputs) {
	output = inputs * weights + biases;
}

double reluActivation(const double& input) {
	if (input < 0) {
		return 0;
	}

	return input;
}

Matrix reluActivationMatrix(const Matrix& m) {
	Matrix r = m;
	for (int i = 0; i < r.rows(); i++) {
		for (int j = 0; j < r.cols(); j++) {
			r[i][j] = reluActivation(r[i][j]);
		}
	}

	return r;
}
