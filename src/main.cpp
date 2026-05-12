#include <chrono>
#include <iostream>
#include <numeric>
#include <random>
#include "math/linAlg.hpp"
#include "network/layer.hpp"
#include "data/dump.hpp"


int main(int argc, char *argv[]) {
	if (argc != 3) {
		std::cout << "Usage: ./program [num_epochs] [batch_size]" << std::endl;
		exit(-1);
	}

    Data trainingData = inputData("./training/mnist_train.csv");
    trainingData.info = trainingData.info * (1.0f / 255.0f);

    Data testingData = inputData("./testing/mnist_test.csv");
    testingData.info = testingData.info * (1.0f / 255.0f);

    int epochs = std::stoi(argv[1]);
    int batchSize = std::stoi(argv[2]);
	int numBatches = trainingData.info.rows() / batchSize;

	
	std::cout << "Preforming " << epochs << " epochs with " << numBatches << " batches of size " << batchSize << "." << std::endl;

	NeuralNetwork nn({784, 128, 64, 10});

    int m = trainingData.info.rows();
    
    srand(time(NULL));
	std::random_device rd;
    std::default_random_engine rng(rd());

    for (int epoch = 0; epoch < epochs; epoch++) {
        std::vector<int> indices(m);
        std::iota(indices.begin(), indices.end(), 0);
        std::shuffle(indices.begin(), indices.end(), rng);

        for (int b = 0; b < numBatches; b++) {
            int start = b * batchSize;

            Matrix batchX(batchSize, trainingData.info.cols());
            Vec batchY(batchSize);
            for (int i = 0; i < batchSize; i++) {
                batchX[i] = trainingData.info[indices[start + i]];
                batchY[i] = trainingData.target[indices[start + i]];
            }
			
            Matrix y = oneHotBatch(batchY, 10);

            // Forward
            nn.step(batchX, y, 10);
			std::cout << "\rEpoch: " << epoch + 1 << " Progress: " << b << " / " << numBatches << std::flush;
        }

		std::cout << std::endl;
    }

	double acc = nn.test(testingData);
	const auto p1 = std::chrono::system_clock::now();
	int time = std::chrono::duration_cast<std::chrono::seconds>( p1.time_since_epoch()).count();
	dumpNN("./models/" + std::to_string(acc) + "-" + std::to_string(time) + ".dat", nn);

    return 0;
}
