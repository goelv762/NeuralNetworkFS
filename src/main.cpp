#include <iostream>
#include <numeric>
#include <random>
#include "math/linAlg.hpp"
#include "network/layer.hpp"

int main(int argc, char *argv[]) {
	if (argc != 4) {
		std::cout << "Usage: ./program [num_epochs] [num_batches] [batch size]" << std::endl;
		exit(-1);
	}
	float lr = 0.1f;
    float lambda = 0.0001f;
    int epochs = std::stoi(argv[1]);
    int numBatches = std::stoi(argv[2]);
    int batchSize = std::stoi(argv[3]);
	
	std::cout << "Preforming " << epochs << " epochs with " << numBatches << " batches of size " << batchSize << "." << std::endl;

    srand(time(NULL));

    Data trainingData = inputData("./training/mnist_train.csv");
    trainingData.info = trainingData.info * (1.0f / 255.0f);

    Data testingData = inputData("./testing/mnist_test.csv");
    testingData.info = testingData.info * (1.0f / 255.0f);

	NeuralNetwork nn({784, 128, 64, 10});

    int m = trainingData.info.rows();
    
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

        }
    }

	nn.test(testingData);

    return 0;
}
