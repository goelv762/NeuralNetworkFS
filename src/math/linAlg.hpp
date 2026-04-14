#pragma once

#include <vector>

struct vec {
	std::vector<double> data;

    vec() = default;
	vec(size_t size) : data(size) {}
    vec(std::initializer_list<double> init) : data(init) {}

    double& operator[](size_t i) { return data[i]; }
    const double& operator[](size_t i) const { return data[i]; }

    size_t size() const { return data.size(); }

    auto begin() { return data.begin(); }
    auto end()   { return data.end(); }

    auto begin() const { return data.begin(); }
    auto end()   const { return data.end(); }
};

using Matrix = std::vector<vec>;

vec operator+(const vec& v1, const vec& v2);
double operator*(const vec& v1, const vec& v2);

std::vector<vec> transpose(const Matrix& m);
vec matrixMultiplication(const vec& v, const Matrix& m);
