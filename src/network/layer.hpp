#include "../math/linAlg.hpp"

class DenseLayer {
	public:

	Matrix weights;
	Vec biases;
	Matrix output;

	DenseLayer(int inputs, int neurons);
	void forward(const Matrix& inputs);
};

double reluActivation(const double& input);

