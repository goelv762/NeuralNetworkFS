#include "../math/linAlg.hpp"

class DenseLayer {
	public:

	Matrix weights;
	Vec biases;
	Matrix output;

	DenseLayer(int inputs, int neurons);
	Matrix forward(const Matrix& inputs);
	double calculate(const Matrix& output, const Vec& truth);
};

Matrix reluActivation(const Matrix& m);
Matrix softmaxActivation(const Matrix& m);

Vec lossCCE(const Matrix& m, const Vec& truth);
Vec accurate(const Matrix& m, const Vec& truth);

