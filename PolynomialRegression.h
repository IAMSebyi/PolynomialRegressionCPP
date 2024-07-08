#pragma once

#include <iostream>
#include <vector>
#include <cmath>

class PolynomialRegression
{
public:
	// Constructor
	PolynomialRegression(const std::vector<std::vector<float>>& features, const std::vector<float>& targets, const int& numOfFeatures, const int& numOfDataPoints, const int& order);

	// Methods
	float Predict(std::vector<float> input, const bool test = false) const;
	void Train(const float& learningRate, const int maxIterations = 40000);
	std::vector<std::vector<float>> GetParameters() const;

private:
	// Methods
	float Cost() const;
	std::vector<float> CoeffGradient(const int& order) const;
	float InterceptGradient() const;
	std::vector<float> GetVectorExponentiation(const std::vector<float>& vec, const float& power) const;
	float GetVectorMultiplication(const std::vector<float>& a, const std::vector<float>& b) const;
	std::vector<float> GetScalarMultiplication(const std::vector<float>& vec, const float& scalar) const;

	// Data members
	std::vector<std::vector<float>> features;
	std::vector<float> targets;

	// Model parameters
	std::vector<std::vector<float>> coefficients;
	float intercept;

	// Variables
	int order;
	int numOfFeatures;
	int numOfDataPoints;
};

