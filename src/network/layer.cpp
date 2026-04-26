#include "layer.hpp"
#include "../math/linAlg.hpp"
#include <cmath>
#include <cstdint>
#include <iostream>

DenseLayer::DenseLayer(int inputs, int neurons) : biases(neurons), dBiases(neurons) {
	weights =  0.1f * randomMatrix(inputs, neurons);
}

Matrix DenseLayer::forward(const Matrix& inputs) {
	input = inputs;
	z = inputs * weights + biases;
	return z;
}

NeuralNetwork::NeuralNetwork(std::vector<int> neuronsVector) {
	for (size_t i = 0; i < neuronsVector.size() - 1; i++) {
		layers.emplace_back(neuronsVector[i], neuronsVector[i + 1]);
	}

	layerCount = layers.size();
}

void NeuralNetwork::step(const Matrix& inputs, const Matrix& y, uint32_t batchSize) {
	forward(inputs);
	backprop(y);
	gradient(inputs, batchSize);
	update();
}

void NeuralNetwork::forward(const Matrix& inputs) {
	layers[0].z = layers[0].forward(inputs);
	layers[0].output = reluActivation(layers[0].z);

	for (size_t i = 1; i < layers.size() - 1; i++) {
		layers[i].z = layers[i].forward(layers[i - 1].output);
		layers[i].output = reluActivation(layers[i].z);
	}

	size_t lastIndex = layerCount - 1;
	layers[lastIndex].z = layers[lastIndex].forward(layers[lastIndex - 1].output);
	layers[lastIndex].output = softmaxActivation(layers[lastIndex].z);
}

void NeuralNetwork::backprop(const Matrix& y) {
	size_t lastIndex = layerCount - 1;
	layers[lastIndex].delta = layers[lastIndex].output - y;
	// use int here becuase size_t is unsingned
	for (int i = lastIndex - 1; i >= 0; i--) {
		layers[i].delta = hadamard(layers[i + 1].delta * transpose(layers[i + 1].weights), reluDerivative(layers[i].z));
	}
}

void NeuralNetwork::gradient(const Matrix& inputs, uint32_t batchSize) {
	size_t lastIndex = layerCount - 1;
	// use int here becuase size_t is unsingned
	for (int i = lastIndex; i > 0; i--) {
		layers[i].dWeights = transpose(layers[i - 1].output) * layers[i].delta * (1.0f / batchSize);
		layers[i].dBiases  = sumRows(layers[i].delta) * (1.0f / batchSize);
	}

	layers[0].dWeights = transpose(inputs) * layers[0].delta * (1.0f / batchSize);
	layers[0].dBiases  = sumRows(layers[0].delta) * (1.0f / batchSize);
}

void NeuralNetwork::update() {
	for (size_t i = 0; i < layerCount; i++) {
		layers[i].weights -= lr * (layers[i].dWeights + lambda * layers[i].weights);
		layers[i].biases  -= lr * layers[i].dBiases;
	}
}

void NeuralNetwork::test(const Data& testingData) {
	forward(testingData.info);
	double testAcc = average(accurate(layers[layerCount - 1].output, testingData.target));
	std::cout << "Test Accuracy: " << testAcc * 100 << "%" << std::endl;
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

Matrix reluDerivative(const Matrix& m) { 
	Matrix r = m;

	for (size_t i = 0; i < r.rows(); i++) {
		for (size_t j = 0; j < r.cols(); j++) {
			if (r[i][j] < 0) {
				r[i][j] = 0;
			} else {
				r[i][j] = 1;
			}
		}
	}

	return r;
}

// used for the output layer (converts to probability)
Matrix softmaxActivation(const Matrix& m) {
    Matrix r = m;
    for (size_t i = 0; i < r.rows(); i++) {
        // find row max
        double maxVal = r[i][0];
        for (size_t j = 1; j < r.cols(); j++)
            if (r[i][j] > maxVal) maxVal = r[i][j];

        double sum = 0;
        for (size_t j = 0; j < r.cols(); j++) {
            r[i][j] = std::exp(r[i][j] - maxVal);
            sum += r[i][j];
        }
        for (size_t k = 0; k < r.cols(); k++)
            r[i][k] /= sum;
    }
    return r;
}

Vec lossCCE(const Matrix& m, const Vec& truth) {
	size_t size = truth.size();
	Vec lossValues(size);
	for (size_t i = 0; i < size; i ++) {
		uint64_t index = static_cast<uint64_t>(std::round(truth[i]));
		double p = std::clamp(m[i][index], 1e-7, 1.0 - 1e-7);
		lossValues[i] = -std::log(p);
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

Matrix oneHotBatch(const Vec& truth, const int& highest) {
	Matrix oneHot(truth.size(), highest);
	
	for (size_t i = 0; i < truth.size(); i++) {
		oneHot[i][truth[i]] = 1.0f;
	}

	return oneHot;
}
