#pragma once

#include <algorithm>
#include <ostream>
#include <vector>

struct Vec {
	std::vector<double> info;

    Vec() = default;
	Vec(size_t size) : info(size) {}
    Vec(std::initializer_list<double> init) : info(init) {}

    double& operator[](size_t i) { return info[i]; }
    const double& operator[](size_t i) const { return info[i]; }

    size_t size() const { return info.size(); }
	const double* data() const { return info.data(); } 

    auto begin() { return info.begin(); }
    auto end()   { return info.end(); }

    auto begin() const { return info.begin(); }
    auto end()   const { return info.end(); }

	void push_back(double newData) { info.push_back(newData); }
};


struct Matrix {
    std::vector<Vec> info;

    Matrix() = default;
	Matrix(std::initializer_list<Vec> init) : info(init) {}
    Matrix(size_t rows, size_t cols) : info(rows, Vec(cols)) {}

    Vec& operator[](size_t i) { return info[i]; }
    const Vec& operator[](size_t i) const { return info[i]; }

    size_t rows() const { return info.size(); }
    size_t cols() const { return info.empty() ? 0 : info[0].size(); }

	// assume that all rows are same length (as a matrix should)
	size_t size() const { return info.size() * info[0].size(); }

	auto begin() { return info.begin(); }
    auto end()   { return info.end(); }

    auto begin() const { return info.begin(); }
    auto end()   const { return info.end(); }

	void push_back(Vec newData) { info.push_back(newData); }
};


// vectors
Vec operator+(const Vec& v1, const Vec& v2);
Vec& operator+=(Vec& v1, const Vec& v2);
Vec operator-(const Vec& v1, const Vec& v2);
Vec& operator-=(Vec& v1, const Vec& v2);
Vec operator*(const double& c, const Vec& v);
Vec operator*(const Vec& v, const double& c);
double operator*(const Vec& v1, const Vec& v2);

// matrix
Matrix operator+(const Matrix& m1, const Matrix& m2);
Matrix& operator+=(Matrix& m1, const Matrix& m2);
Matrix operator-(const Matrix& m1, const Matrix& m2);
Matrix& operator-=(Matrix& m1, const Matrix& m2);
Matrix operator*(const Matrix& m1, const Matrix& m2);

// matrix & something mix
Matrix operator+(const Matrix& m, const Vec& v);
Matrix operator+(const Matrix& m, const double& c);
Matrix operator*(const Matrix& m, const double& c);
Matrix operator*(const double& c, const Matrix& m);

// printing
std::ostream& operator<<(std::ostream& os, const Vec& v);
std::ostream& operator<<(std::ostream& os, const Matrix& m);

double sum(const Vec& v);
double average(const Vec& v);
Matrix transpose(const Matrix& m);
Matrix clipMatrix(const Matrix& m, const double& bound);
Matrix randomMatrix(size_t rows, size_t cols);

Vec matrixToVec(const Matrix& m);

Vec sumRows(const Matrix& m);
Matrix hadamard(const Matrix& m1, const Matrix& m2);
