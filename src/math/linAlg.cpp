#include "linAlg.hpp"

vec operator+(const vec& v1, const vec& v2) {
	vec v3(v1.size());

	for (size_t i = 0; i < v1.size(); i++) {
		v3[i] = v1[i] + v2[i];
	}

	return v3;
}

double operator*(const vec& v1, const vec& v2) {
	double dot;

	for (size_t i = 0; i < v1.size(); i++) {
		dot += v1[i] * v2[i];
	}

	return dot;
}

std::vector<vec> transpose(const std::vector<vec>& m) {
    size_t rows = m.size();
   	size_t cols = m[0].size();

    std::vector<vec> t(cols, vec(rows));

    for (size_t i = 0; i < rows; i++) {
        for (size_t j = 0; j < cols; j++) {
            t[j][i] = m[i][j];
        }
    }

    return t;
}


vec matrixMultiplication(const vec& v, const Matrix& m) {
	size_t size = m.size();

	vec result(size);

	for (size_t i = 0; i < size; i++) {
		result[i] = v * m[i];
	}

	return result;
}
