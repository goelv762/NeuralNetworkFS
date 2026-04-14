#include "../math/linAlg.hpp"

class DenseLayer {
	public:

	Matrix weights;
	Vec biases;
	Matrix output;

	DenseLayer(int inputs, int neurons);
	void forward(const Matrix& inputs);
};

Matrix reluActivation(const Matrix& m);
Matrix softmaxActivation(const Matrix& m);

