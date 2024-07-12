# Polynomial Regression Model in C++

This project implements a polynomial regression model with multiple input features in C++ for educational purposes. It includes methods to train the model using gradient descent, predict outputs for given inputs, and calculate the model parameters.

## Features

- Train a polynomial regression model using gradient descent.
- Predict target values for new input data.
- Calculate and output model parameters (coefficients and intercept).
- Handle multiple input features and data points.

## Formulas

### Model Function ($n$ degree polynomial)

$$f_{\vec{w},b}(\vec{x}) = b + \vec{w_1} * \vec{x} + \vec{w_2} * \vec{x^2} + ... + \vec{w_1} * \vec{x^n} $$ where $\vec{w_k}$ is the $k^{th}$ order coefficients vector, $b$ is the intercept and $\vec{x}$ is the input features vector

### Squared Error Cost Function

$$J(\vec{w}, b) = \frac{1}{2m}	\sum_{i=1}^m (\hat{y}^{(i)} - y^{(i)})^2$$ where $m$ is the number of training examples, $\hat{y}^{(i)}$ is the predicted output target for the $i^{th}$ training data point and $y^{(i)}$ is the real target value

### Gradient with respect to the $n^{th}$ order coefficients vector parameter

$$\frac{\delta}{\delta \vec{w}}J(\vec{w}, b) = \frac{1}{m}\sum_{i=1}^m(f_{\vec{w}, b}(\vec{x}^{(i)})-y^{(i)})*(\vec{x}^{(i)})^n$$ where $\vec{x}^{(i)}$ is the input features vector of the $i^{th}$ training data point

### Gradient with respect to the intercept parameter

$$\frac{\delta}{\delta b}J(\vec{w}, b) = \frac{1}{m}\sum_{i=1}^m(f_{\vec{w}, b}(\vec{x}^{(i)})-y^{(i)})$$
