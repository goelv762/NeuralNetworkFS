#pragma once

#include "../math/linAlg.hpp"
#include "../data/csv.hpp"
#include <cstdint>

constexpr double lr = 0.1f;
constexpr double lambda = 0.0001f;

class DenseLayer {
	public:

	Matrix weights;
    Vec biases;

    Matrix input;   // a ^ l-1
    Matrix z; 		// pre activation
    Matrix output;  // a ^ l
    Matrix delta;   // error

    Matrix dWeights;
    Vec dBiases;

	DenseLayer(int inputs, int neurons);
	Matrix forward(const Matrix& inputs);
	double calculate(const Matrix& output, const Vec& truth);
};

class NeuralNetwork {
	public:
	std::vector<int> nv; // for dumping
	std::vector<DenseLayer> layers;

	NeuralNetwork(std::vector<int> neuronsVector);
	void step(const Matrix& inputs, const Matrix& y, uint32_t batchSize);
	double test(const Data& testingData);

	size_t layerCount;

	void forward(const Matrix& inputs);
	void backprop(const Matrix& y);
	void gradient(const Matrix& inputs, uint32_t batchSize);
	void update();
};

struct layerData {
	Matrix weights;
	Vec biases;
};

Matrix reluActivation(const Matrix& m);
Matrix reluDerivative(const Matrix& m);
Matrix softmaxActivation(const Matrix& m);

Vec lossCCE(const Matrix& m, const Vec& truth);
Vec accurate(const Matrix& m, const Vec& truth);

Matrix oneHotBatch(const Vec& truth, const int& highest);
