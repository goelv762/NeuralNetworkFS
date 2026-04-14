#include <iostream>
#include <vector>

struct Neuron {
	std::vector<double> weights;
	double bias;
};

std::vector<double> neuronOutputs(
	std::vector<double> inputs,
	std::vector<Neuron> layer
) {
	std::vector<double> outputs;
	for (Neuron& n : layer) {
		double result = 0.0f;
		// assume input length and weights length is the same
		for (int i = 0; i < inputs.size(); i++) {
			result += inputs[i] * n.weights[i];
		}
		
		result += n.bias;
		outputs.push_back(result);
	}


	return outputs;
}

int main (int argc, char *argv[]) {
	std::vector<double> inputs = {1.0f, 2.0f, 3.0f, 2.5f};
	Neuron n1 = {
		{0.2, 0.8, -0.5, 1},
		2.0f
	};

	Neuron n2 = {
		{0.5, -0.91, 0.26, -0.5},
		3.0f
	};

	Neuron n3 = {
		{-0.26, -0.27, 0.17, 0.87},
		0.5f
	};
	
	for (double& out : neuronOutputs(inputs, {n1, n2, n3})) {
		std::cout << out << " " << std::endl;
	}
	
	return 0;
}
