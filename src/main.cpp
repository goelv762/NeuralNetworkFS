#include <iostream>

#include "math/linAlg.hpp"

struct Layer {
	Matrix W; // weight table
	Vec b; // biases
};

Matrix neuronOutputs(Matrix& inputs, Layer& layer) {
	return inputs * transpose(layer.W) + layer.b;
}

int main (int argc, char *argv[]) {
	// feature set
	Matrix inputs = {
		{1, 2, 3, 2.5}, 
		{2, 5, -1, 2}, 
		{-1.5, 2.7, 3.3, -0.8}
	};

	Layer layer = {
		// weights
		{{0.2, 0.8, -0.5, 1},
		{0.5, -0.91, 0.26, -0.5},
		{-0.26, -0.27, 0.17, 0.87}},
		// biases
		{2.0f, 3.0f, 0.5f} 
	};
	
	Matrix outputs = neuronOutputs(inputs, layer);
	for (Vec& outVec : outputs) {
		for (double& out : outVec) {
			std::cout << out << "\t";
		}

		std::cout << std::endl;
	}
	
	return 0;
}
