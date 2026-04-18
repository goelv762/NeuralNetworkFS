#include "layer.hpp"
#include "../math/linAlg.hpp"
#include <cmath>
#include <cstdint>
#include <iostream>

DenseLayer::DenseLayer(int inputs, int neurons) : biases(neurons) {
	weights = 0.01 * randomMatrix(inputs, neurons);
}

Matrix DenseLayer::forward(const Matrix& inputs) {
	Matrix clipped = clipMatrix(inputs, std::numeric_limits<double>::epsilon());
	return clipped * weights + biases;
}

// used for the hidden layers
Matrix reluActivation(const Matrix& m) {
	Matrix r = m;
	for (size_t i = 0; i < r.rows(); i++) {
		for (size_t j = 0; j < r.cols(); j++) {
			if (r[i][j] < 0) {
				r[i][j] = 0;
			}
		}
	}

	return r;
}

// used for the output layer (converts to probability)
Matrix softmaxActivation(const Matrix& m) {
	Matrix r = m;

	for (size_t i = 0; i < r.rows(); i++) {
		double sum = 0;
		for (size_t j = 0; j < r.cols(); j++) {
			r[i][j] = std::exp(r[i][j]);
			sum += r[i][j];
		}

		for (int k = 0; k < r.cols(); k++) {
			r[i][k] /= sum;
		}
	}

	return r;
}

Vec lossCCE(const Matrix& m, const Vec& truth) {
	size_t size = truth.size();
	Vec lossValues(size);
	for (size_t i = 0; i < size; i ++) {
		uint64_t index = static_cast<uint64_t>(std::round(truth[i]));
		lossValues[i] = -std::log(m[i][index]);
	}

	return lossValues;
}

Vec accurate(const Matrix& m, const Vec& truth) {
	size_t size = m.rows();
	Vec accuracy(size);

	for (size_t i = 0; i < size; i ++) {
		uint64_t index = static_cast<uint64_t>(std::round(truth[i]));
		// get highest confidence
		int highestIndex = 0;
		double highestConfidence = m[i][0];
		for (size_t j = 0; j < m.cols(); j++) {
			if (m[i][j] > highestConfidence) {
				highestIndex = j;
				highestConfidence = m[i][j];
			}
		}

		accuracy[i] = (index == highestIndex);
	}

	return accuracy;
}
