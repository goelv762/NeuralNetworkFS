#include <cstdint>
#include <iostream>
#include <random>

#include "math/linAlg.hpp"
#include "network/layer.hpp"

constexpr uint32_t inputs = 2;
constexpr uint32_t neurons = 3;

int main (int argc, char *argv[]) {

	DenseLayer dl(inputs, neurons);

	Matrix dataInputs = {
		{0.00000, 0.00000},
		{0.00300, 0.00965},
		{0.01288, 0.01556},
		{0.02997, 0.00445},
		{0.03931, 0.00933}
	};

	dl.forward(dataInputs);

	std::cout << dl.output << std::endl;
	
	return 0;
}
