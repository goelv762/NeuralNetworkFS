#include "dump.hpp"
#include "../network/layer.hpp"
#include <filesystem>
#include <fstream>
#include <iostream>
#include <ranges>

void dumpNN(std::string path, const NeuralNetwork& nn) {
    std::string fullPath = std::filesystem::absolute(path).string();
    std::ofstream outFile(fullPath, std::ios::binary);
	
	std::string header = "";
	for (size_t i = 0; i < nn.nv.size(); i++) {
		header += std::to_string(nn.nv[i]);
		// dont add space at the end
		header += i == nn.nv.size() - 1 ? "" : " ";
	}

	header += "\n";

    outFile.write(header.c_str(), header.size());
	
	for (const DenseLayer& layer : nn.layers) {
		for (const Vec& v : layer.weights) {
			outFile.write(reinterpret_cast<const char*>(v.data()), v.size() * sizeof(double));
		}

		outFile.write(reinterpret_cast<const char*>(layer.biases.data()), layer.biases.size() * sizeof(double));
	}

    outFile.close();
}

NeuralNetwork readNN(std::string path) {
	std::string fullPath = std::filesystem::absolute(path).string();
	std::fstream inFile(fullPath, std::ios::in | std::ios::binary);
	
	std::string header;
	std::getline(inFile, header);
    std::string delim = " ";

    auto words = header | std::views::split(delim);
	std::vector<int> neuronCounts;

    for (auto word : words) {
		std::string curr = static_cast<std::string>(std::string_view(word.begin(), word.end()));
		neuronCounts.push_back(std::stoi(curr));
    }

	NeuralNetwork nn(neuronCounts);
	
	for (size_t i = 0; i < neuronCounts.size() - 1; i++) {
		size_t flatSizeWeights = neuronCounts[i] * neuronCounts[i + 1];
		double bufferWeights[flatSizeWeights];
		inFile.read(reinterpret_cast<char*>(bufferWeights), sizeof(bufferWeights));

		Matrix weights(neuronCounts[i], neuronCounts[i + 1]);
		for (size_t j = 0; j < flatSizeWeights; j++) {
			weights[j / neuronCounts[i + 1]][j % neuronCounts[i + 1]] = bufferWeights[j];
		}

		size_t flatSizeBiases = neuronCounts[i + 1];
		double bufferBiases[flatSizeBiases];
		inFile.read(reinterpret_cast<char*>(bufferBiases), sizeof(bufferBiases));
		
		Vec biases(neuronCounts[i + 1]);
		for (size_t k = 0; k < flatSizeBiases; k++) {
			biases[k] = bufferBiases[k];
		}

		nn.layers[i].weights = weights;
		nn.layers[i].biases = biases;
	}


	return nn;
}
