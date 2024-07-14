#include "PolynomialRegression.h"

// Constructor to initialize the PolynomialRegression object with features, targets, and dimensions
PolynomialRegression::PolynomialRegression(const std::vector<std::vector<float>>& features, const std::vector<float>& targets, const int& numOfFeatures, const int& numOfDataPoints, const int& order, const float& regularizationParam)
	: features(features), targets(targets), numOfFeatures(numOfFeatures), numOfDataPoints(numOfDataPoints), coefficients(order, std::vector<float>(numOfFeatures, 0)), intercept(0), order(order), regularizationParam(regularizationParam)
{
}

// Predict the target value for a given input vector
float PolynomialRegression::Predict(std::vector<float> input) const
{
	float result = intercept + GetVectorMultiplication(input, coefficients[0]);
	for (int i = 2; i <= order; i++) {
		std::vector<float> term = GetVectorExponentiation(input, i);
		result += GetVectorMultiplication(term, coefficients[i-1]);
	}

	return result;
}

// Calculate the squared error cost function with L2 regularization
float PolynomialRegression::Cost() const
{
	float result = 0;

	for (int i = 0; i < numOfDataPoints; i++) {
		float error = Predict(features[i]) - targets[i];
		result += error * error;
	}

	for (int i = 0; i < order; i++) {
		result += regularizationParam * GetVectorMultiplication(coefficients[i], coefficients[i]);
	}

	result /= 2 * numOfDataPoints;

	return result;
}

// Calculate the gradient of the cost function with respect to the coefficients vector parameter of a certain order
std::vector<float> PolynomialRegression::CoeffGradient(const int& order) const
{
	std::vector<float> result (numOfFeatures, 0);

	for (int i = 0; i < numOfFeatures; i++) {
		for (int j = 0; j < numOfDataPoints; j++) {
			result[i] += (Predict(features[j]) - targets[j]) * std::pow(features[j][i], order);
		}
		result[i] += regularizationParam * coefficients[order - 1][i];
		result[i] /= numOfDataPoints;
	}

	return result;
}

// Calculate the gradient of the cost function with respect to intercept parameter
float PolynomialRegression::InterceptGradient() const
{
	float result = 0;

	for (int i = 0; i < numOfDataPoints; i++) {
		result += Predict(features[i]) - targets[i];
	}
	result /= numOfDataPoints;

	return result;

}

// Compute vector with elements raised to a certian power
std::vector<float> PolynomialRegression::GetVectorExponentiation(const std::vector<float>& vec, const float& power) const
{
	std::vector<float> result;
	for (const auto& val : vec) { result.push_back(std::pow(val, power)); }
	return result;
}

// Calculate the multiplication of two vectors
float PolynomialRegression::GetVectorMultiplication(const std::vector<float>& a, const std::vector<float>& b) const
{
	float result = 0;
	int i = 0;
	for (const auto& val : a) {
		result += val * b[i];
		i++;
	}
	return result;
}

// Calculate the multiplication of a vector by a scalar
std::vector<float> PolynomialRegression::GetScalarMultiplication(const std::vector<float>& vec, const float& scalar) const
{
	std::vector<float> result;
	for (const auto& val : vec) { result.push_back(val * scalar); }
	return result;
}

// Train the model using Gradient Descent
void PolynomialRegression::Train(const float& learningRate, const int maxIterations)
{
	const float convergenceThreshold = 1e-20; // Convergence threshold
	const int logStep = 1000; // Logging step interval
	
	int i;
	for (i = 1; i <= maxIterations; i++) {
		float prevCost = Cost();
		std::vector<std::vector<float>> prevCoefficients = coefficients;
		float prevIntercept = intercept;

		for (int i = 1; i <= order; i++) {
			std::vector<float> coeffSlope = CoeffGradient(i);
			for (int j = 0; j < numOfFeatures; j++) {
				coefficients[i - 1][j] -= learningRate * coeffSlope[j];
			}
		}
		intercept -= learningRate * InterceptGradient();

		float currentCost = Cost();

		// Check for convergence
		if (i > 1 && (currentCost > prevCost || prevCost - currentCost <= convergenceThreshold)) {
			// Early stoppage due to convergence
			coefficients = prevCoefficients;
			intercept = prevIntercept;
			break;
		}

		if (i % logStep == 0) {
			std::cout << "Iteration #" << i << ", Cost: " << currentCost << '\n';
		}
	}

	// Output final cost to console for learning rate adjustments
	std::cout << "Final cost after " << i - 1 << " iterations : " << Cost() << "\n \n";
}

// Get the model parameters (coefficients and intercept)
std::vector<std::vector<float>> PolynomialRegression::GetParameters() const
{
	std::vector<std::vector<float>> parameters(coefficients);
	parameters.push_back(std::vector<float>(1, intercept));
	return parameters;
}
