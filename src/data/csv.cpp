#include "csv.hpp"
#include <string>
#include <filesystem>
#include <fstream>
#include <sstream>

Data inputData(std::string path) {
	Matrix data;

	std::string fullPath = std::filesystem::absolute(path).string();
	
	std::ifstream file(fullPath);
	std::string line;
	
	while (std::getline(file, line)) {
		std::stringstream ss(line);
		std::string value;
		Vec row;

		while (std::getline(ss, value, ',')) {
			row.push_back(stod(value));
		}

		data.push_back(row);
	}

	if (data.rows() == 0 && data.cols() == 0) {
		throw std::runtime_error("could not read: " + fullPath);
	}

	file.close();

	size_t rows = data.rows();
	size_t cols = data.cols();

	Matrix info(data.rows(), data.cols() - 1);
	Vec target(data.rows());

    for (size_t i = 0; i < rows; i++) {
        for (size_t j = 0; j < cols; j++) {
			if (j < cols - 1) {
				info[i][j] = data[i][j];
			} else {
				target[i] = data[i][j];
			}
		}
	}
	return {info, target};
}
