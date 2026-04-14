#pragma once

#include <ostream>
#include <vector>

struct Vec {
	std::vector<double> data;

    Vec() = default;
	Vec(size_t size) : data(size) {}
    Vec(std::initializer_list<double> init) : data(init) {}

    double& operator[](size_t i) { return data[i]; }
    const double& operator[](size_t i) const { return data[i]; }

    size_t size() const { return data.size(); }

    auto begin() { return data.begin(); }
    auto end()   { return data.end(); }

    auto begin() const { return data.begin(); }
    auto end()   const { return data.end(); }
};


struct Matrix {
    std::vector<Vec> data;

    Matrix() = default;
	Matrix(std::initializer_list<Vec> init) : data(init) {}
    Matrix(size_t rows, size_t cols) : data(rows, Vec(cols)) {}

    Vec& operator[](size_t i) { return data[i]; }
    const Vec& operator[](size_t i) const { return data[i]; }

    size_t rows() const { return data.size(); }
    size_t cols() const { return data.empty() ? 0 : data[0].size(); }

	auto begin() { return data.begin(); }
    auto end()   { return data.end(); }

    auto begin() const { return data.begin(); }
    auto end()   const { return data.end(); }
};

// vectors
Vec operator+(const Vec& v1, const Vec& v2);
Vec& operator+=(Vec& v1, const Vec& v2);
Vec operator-(const Vec& v1, const Vec& v2);
Vec& operator-=(Vec& v1, const Vec& v2);
double operator*(const Vec& v1, const Vec& v2);

// matrix
Matrix operator+(const Matrix& m1, const Matrix& m2);
Matrix operator*(const Matrix& m1, const Matrix& m2);

// matrix & something mix
Matrix operator+(const Matrix& m, const Vec& v);
Matrix operator*(const Matrix& m, const double& c);
Matrix operator*(const double& c, const Matrix& m);

// printing
std::ostream& operator<<(std::ostream& os, const Vec& v);
std::ostream& operator<<(std::ostream& os, const Matrix& m);


Matrix transpose(const Matrix& m);
Matrix randomMatrix(size_t rows, size_t cols);
