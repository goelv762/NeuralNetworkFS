#include <cstdint>
#include <iostream>
#include <random>

#include "math/linAlg.hpp"
#include "network/layer.hpp"

const Matrix dataInputs = {
	{0.00000, 0.00000},
	{0.00300, 0.00965},
	{0.01288, 0.01556},
	{0.02997, 0.00445},
	{0.03931, 0.00933}
};

int main (int argc, char *argv[]) {

	DenseLayer l1(2, 3);
	l1.forward(dataInputs);
	l1.output = reluActivation(l1.output);

	DenseLayer l2(3, 3);
	l2.forward(l1.output);
	l2.output = softmaxActivation(l2.output);

	std::cout << l1.output << std::endl;
	std::cout << l2.output << std::endl;
	
	return 0;
}
