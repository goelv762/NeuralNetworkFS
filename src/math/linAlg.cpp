#include "linAlg.hpp"
#include <algorithm>
#include <cstdio>
#include <iomanip>
#include <iostream>

Vec operator+(const Vec& v1, const Vec& v2) {
	Vec v3(v1.size());

	for (size_t i = 0; i < v1.size(); i++) {
		v3[i] = v1[i] + v2[i];
	}

	return v3;
}

Vec& operator+=(Vec& v1, const Vec& v2) {
	v1 = v1 + v2;
	return v1;
}

Vec operator-(const Vec& v1, const Vec& v2) {
	Vec v3(v1.size());

	for (size_t i = 0; i < v1.size(); i++) {
		v3[i] = v1[i] - v2[i];
	}

	return v3;
}

Vec& operator-=(Vec& v1, const Vec& v2) {
	v1 = v1 - v2;
	return v1;
}

Vec operator*(const double& c, const Vec& v) {
	Vec r(v.size());

	for (size_t i = 0; i < v.size(); i++) {
		r[i] = c * v[i];
	}

	return r;
}

Vec operator*(const Vec& v, const double& c) {
	return c * v;
}

double operator*(const Vec& v1, const Vec& v2) {
	double dot = 0;

	for (size_t i = 0; i < v1.size(); i++) {
		dot += v1[i] * v2[i];
	}

	return dot;
}


Matrix operator*(const Matrix& m1, const Matrix& m2) {
	// assume that m1[0].size() == m2.size() "inner dimensions"
	// NOTE, matrix multiplication is not commutative!
	
	size_t rows = m1.rows();
	size_t cols = m2.cols();

	Matrix p(rows, cols);

	for (size_t i = 0; i < rows; i++) {
        for (size_t j = 0; j < cols; j++) {
            p[i][j] = 0;

            for (size_t k = 0; k < m1.cols(); k++) {
                p[i][j] += m1[i][k] * m2[k][j];
            }
        }
    }

	return p;
}

Matrix operator+(const Matrix& m1, const Matrix& m2) {
	size_t rows = m1.rows();
	size_t cols = m2.cols();

	Matrix r(rows, cols);

	for (size_t i = 0; i < rows; i++) {
        for (size_t j = 0; j < cols; j++) {
            r[i][j] = m1[i][j] + m2[i][j];
        }
    }

	return r;
}

Matrix& operator+=(Matrix& m1, const Matrix& m2) {
	m1 = m1 + m2;
	return m1;
}

Matrix operator-(const Matrix& m1, const Matrix& m2) {
	size_t rows = m1.rows();
	size_t cols = m2.cols();

	Matrix r(rows, cols);

	for (size_t i = 0; i < rows; i++) {
        for (size_t j = 0; j < cols; j++) {
            r[i][j] = m1[i][j] - m2[i][j];
        }
    }

	return r;
}

Matrix& operator-=(Matrix& m1, const Matrix& m2) {
	m1 = m1 - m2;
	return m1;
}

Matrix operator+(const Matrix& m, const Vec& v) {
	// broadcasting ?
	Matrix r = m;
	for (Vec& row : r) {
		row += v;
	}

	return r;
}

Matrix operator+(const Matrix& m, const double& c) {
	Matrix r = m;
	for (Vec& row : r) {
		for (double& val : row) {
			val += c;
		}
	}

	return r;
}

Matrix operator*(const Matrix& m, const double& c) {
	Matrix r = m;
	size_t rows = m.rows();
	size_t cols = m.cols();

	for (size_t i = 0; i < rows; i++) {
        for (size_t j = 0; j < cols; j++) {
            r[i][j] *= c;
        }
    }

	return r;
}

Matrix operator*(const double& c, const Matrix& m) {
	return m * c;
}

std::ostream& operator<<(std::ostream& os, const Vec& v) {
	for (const double& o : v) {
		os << std::fixed << std::setprecision(5) << o << "\t";
	}

	os << std::endl;

	return os;
}

std::ostream& operator<<(std::ostream& os, const Matrix& m) {
	for (const Vec& v : m) {
		os << v;
	}

	return os;
}

double sum(const Vec& v) {
	double vecSum = 0;
	for (const double& value : v) {
		vecSum += value;
	}

	return vecSum;
}

double average(const Vec& v) {
	return sum(v) / v.size();
}

Matrix transpose(const Matrix& m) {
    size_t rows = m.rows();
   	size_t cols = m.cols();

    Matrix t(cols, rows);

    for (size_t i = 0; i < rows; i++) {
        for (size_t j = 0; j < cols; j++) {
            t[j][i] = m[i][j];
        }
    }

    return t;
}

Matrix clipMatrix(const Matrix& m, const double& bound) {
	Matrix r = m;
    for (size_t i = 0; i < m.rows(); i++) {
        for (size_t j = 0; j < m.cols(); j++) {
			r[i][j] = std::clamp(r[i][j], bound, 1 - bound);
		}
	}

	return r;
}


Matrix randomMatrix(size_t rows, size_t cols) {
	Matrix r(rows, cols);

    for (size_t i = 0; i < rows; i++) {
        for (size_t j = 0; j < cols; j++) {
			// generate random number between 0 and 2
			// subtract 1 to get between -1 and 1
            r[i][j] = (double)rand() / (double)(RAND_MAX / 2) - 1.0f;
        }
    }

	return r;
}

Vec matrixToVec(const Matrix& m) {
	return m[0];
}

Vec sumRows(const Matrix& m) {
	Vec v(m.cols());
	for (size_t i = 0; i < m.cols(); i++) {
		v[i] = 0.0f;
		for (size_t j = 0; j < m.rows(); j++) {
			v[i] += m[j][i];
		}
	}

	return v;
}

Matrix hadamard(const Matrix& m1, const Matrix& m2) {
    Matrix r(m1.rows(), m2.cols());

    for (size_t i = 0; i < m1.rows(); i++) {
        for (size_t j = 0; j < m1.cols(); j++) {
            r[i][j] = m1[i][j] * m2[i][j];
        }
    }

    return r;
}
