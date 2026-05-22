#include <chrono>
#include <filesystem>
#include <iostream>
#include <numeric>
#include <random>
#include "math/linAlg.hpp"
#include "network/layer.hpp"
#include "data/dump.hpp"


int main(int argc, char *argv[]) {

	if (argc != 3) {
		std::cout << "To train new model, use './program [num_epochs] [batch_size]'" << std::endl;

		std::string modelPath = "./models/";
		// assume a model wants to be selected for usage
		int index = 0;
		std::vector<std::string> modelPaths;
		for (const auto& modelFile : std::filesystem::directory_iterator(modelPath)) {
			// output to user
            std::cout << "[" << index << "]: "<< modelFile.path().filename() << std::endl;
			// save for option choice by user
			modelPaths.push_back(modelFile.path().filename());

			index ++;
        }

		std::cout << "Select a model: ";
		
		int modelIndex;
		std::cin >> modelIndex;

		NeuralNetwork nn = readNN(modelPath + modelPaths[modelIndex]);

		Data testingData = inputData("./testing/mnist_test.csv");
		testingData.info = testingData.info * (1.0f / 255.0f);

		nn.test(testingData);
	}
	
	if (argc == 3) {
		Data trainingData = inputData("./training/mnist_train.csv");
		trainingData.info = trainingData.info * (1.0f / 255.0f);

		Data testingData = inputData("./testing/mnist_test.csv");
		testingData.info = testingData.info * (1.0f / 255.0f);

		int epochs = std::stoi(argv[1]);
		int batchSize = std::stoi(argv[2]);
		int numBatches = trainingData.info.rows() / batchSize;

		
		std::cout << "Preforming " << epochs << " epochs with " << numBatches << " batches of size " << batchSize << "." << std::endl;
		
		// TODO: allow user to specify 
		NeuralNetwork nn({784, 128, 64, 10});

		int m = trainingData.info.rows();
		
		srand(time(NULL));
		std::random_device rd;
		std::default_random_engine rng(rd());

		for (int epoch = 0; epoch < epochs; epoch++) {
			std::vector<int> indices(m);
			std::iota(indices.begin(), indices.end(), 0);
			std::shuffle(indices.begin(), indices.end(), rng);

			for (int batch = 0; batch < numBatches; batch++) {
				int start = batch * batchSize;

				Matrix batchX(batchSize, trainingData.info.cols());
				Vec batchY(batchSize);
				for (int i = 0; i < batchSize; i++) {
					batchX[i] = trainingData.info[indices[start + i]];
					batchY[i] = trainingData.target[indices[start + i]];
				}
				
				Matrix y = oneHotBatch(batchY, 10);

				// Forward
				nn.step(batchX, y, 10);
				std::cout << "\rEpoch: " << epoch + 1 << " Progress: " << batch << " / " << numBatches << std::flush;
			}

			std::cout << std::endl;
		}
		
		// test the model
		double acc = nn.test(testingData);

		// save model with filename of time it was generated + accuracy
		const auto p1 = std::chrono::system_clock::now();
		int time = std::chrono::duration_cast<std::chrono::seconds>( p1.time_since_epoch()).count();
		dumpNN("./models/" + std::to_string(acc) + "-" + std::to_string(time) + ".dat", nn);
	}


    return 0;
}
