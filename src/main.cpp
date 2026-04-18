#include <iostream>

#include "math/linAlg.hpp"
#include "network/layer.hpp"
#include "data/csv.hpp"

int main (int argc, char *argv[]) {

	Data data = inputData("data/spiral.csv");
	
	DenseLayer l1(2, 3);
	l1.output = l1.forward(data.info);
	l1.output = reluActivation(l1.output);

	DenseLayer l2(3, 3);
	l2.output = l2.forward(l1.output);
	l2.output = softmaxActivation(l2.output);

	// std::cout << average(lossCCE(l2.output, data.target)) << std::endl;
	std::cout << average(accurate(l2.output, data.target)) << std::endl;

	return 0;
}
