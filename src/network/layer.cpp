#include "layer.hpp"
#include "../math/linAlg.hpp"
#include <cmath>

DenseLayer::DenseLayer(int inputs, int neurons) : biases(neurons) {
	weights = 0.01 * randomMatrix(inputs, neurons);
}

void DenseLayer::forward(const Matrix& inputs) {
	output = inputs * weights + biases;
}

Matrix reluActivation(const Matrix& m) {
	Matrix r = m;
	for (int i = 0; i < r.rows(); i++) {
		for (int j = 0; j < r.cols(); j++) {
			if (r[i][j] < 0) {
				r[i][j] = 0;
			}
		}
	}

	return r;
}

Matrix softmaxActivation(const Matrix& m) {
	Matrix r = m;

	for (int i = 0; i < r.rows(); i++) {
		double sum = 0;
		for (int j = 0; j < r.cols(); j++) {
			r[i][j] = std::exp(r[i][j]);
			sum += r[i][j];
		}

		for (int k = 0; k < r.cols(); k++) {
			r[i][k] /= sum;
		}
	}

	return r;
}
