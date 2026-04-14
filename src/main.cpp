#include <iostream>

#include "math/linAlg.hpp"

struct Layer {
	Matrix W; // weight table
	vec b; // biases
};

vec neuronOutputs(vec& inputs, Layer& layer) {
	return matrixMultiplication(inputs, layer.W) + layer.b;
}

int main (int argc, char *argv[]) {
	// feature set
	vec inputs = {1.0f, 2.0f, 3.0f, 2.5f};
	Layer layer = {
		// weights
		{{0.2, 0.8, -0.5, 1},
		{0.5, -0.91, 0.26, -0.5},
		{-0.26, -0.27, 0.17, 0.87}},
		// biases
		{2.0f, 3.0f, 0.5f} 
	};
	
	vec outputs = neuronOutputs(inputs, layer);
	for (double& out : outputs) {
		std::cout << out << " " << std::endl;
	}
	
	return 0;
}
