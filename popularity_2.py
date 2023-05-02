import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.neighbors import KNeighborsRegressor

# ReLU activation function
def relu(x):
    return np.maximum(x, 0)

# Derivative of ReLU activation function
def relu_derivative(x):
    return (x > 0) * 1

# Mean Squared Error loss function
def mse(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

# Derivative of Mean Squared Error loss function
def mse_derivative(y_true, y_pred):
    return 2 * (y_pred - y_true)

class NeuralNet:
    def __init__(self, input_size, hidden_size):
        self.W1 = np.random.randn(input_size, hidden_size)  # Initialize weights for the input layer
        self.b1 = np.zeros(hidden_size)  # Initialize biases for the input layer
        self.W2 = np.random.randn(hidden_size, 1)  # Initialize weights for the hidden layer
        self.b2 = np.zeros(1)  # Initialize biases for the hidden layer

    # Forward pass through the network
    def forward(self, X):
        self.Z1 = np.dot(X, self.W1) + self.b1  # Linear transformation for the input layer
        self.A1 = relu(self.Z1)  # Apply ReLU activation function to the input layer
        self.Z2 = np.dot(self.A1, self.W2) + self.b2  # Linear transformation for the hidden layer
        return self.Z2  # Return the output without applying an activation function for regression

    # Backward pass to compute gradients and update weights and biases
    def backward(self, X, y, y_pred, learning_rate):
        m = X.shape[0]
        dZ2 = mse_derivative(y, y_pred)  # Compute the derivative of the loss function with respect to the output
        dW2 = (1 / m) * np.dot(self.A1.T, dZ2)  # Compute the gradient of the loss function with respect to W2
        db2 = (1 / m) * np.sum(dZ2, axis=0)  # Compute the gradient of the loss function with respect to b2

        dA1 = np.dot(dZ2, self.W2.T)  # Compute the gradient of the loss function with respect to A1
        dZ1 = dA1 * relu_derivative(self.Z1)  # Compute the gradient of the loss function with respect to Z1
        dW1 = (1 / m) * np.dot(X.T, dZ1)  # Compute the gradient of the loss function with respect to W1
        db1 = (1 / m) * np.sum(dZ1, axis=0)  # Compute the gradient of the loss function with respect to b1

        self.W1 -= learning_rate * dW1  # Update weights for the input layer
        self.b1 -= learning_rate * db1  # Update biases for the input layer
        self.W2 -= learning_rate * dW2  # Update weights for the hidden layer
        self.b2 -= learning_rate * db2  # Update biases for the hidden layer

    # Train the neural network using stochastic gradient descent
    # Continue with the train method for the SimpleNN class
    def train(self, X, y, iterations, learning_rate):
        for itr in range(iterations):
            y_pred = self.forward(X)  # Perform a forward pass
            loss = mse(y, y_pred)  # Compute the loss
            self.backward(X, y, y_pred, learning_rate)  # Perform a backward pass and update weights and biases

            if itr % 100 == 0:
                print(f"iteration {itr}, Loss: {loss}")

# Load and preprocess data
files = ["webscraping/Billboard Hot 100.csv", "webscraping/BUTTER.csv", "webscraping/Electric Lady Studios.csv", "webscraping/Fresh Finds.csv", "webscraping/Best of the Decade For You.csv"]
frames = []
for f in files:
    frames.append(pd.read_csv(f, encoding='cp1252'))
df = pd.concat(frames)
# Remove leading and trailing spaces from column names
df.columns = [col.strip() for col in df.columns]
X = df.iloc[:, 1:-1]
X_normalized = (X - X.mean()) / X.std()
y = df["popularity"].values.reshape(-1, 1)

X_train, X_test, y_train, y_test = train_test_split(X_normalized, y, test_size=0.2, random_state=42)

# Train the custom neural network model
input_size = X_train.shape[1]
hidden_size = 16
iterations = 1000
learning_rate = 0.01

nn_model = NeuralNet(input_size, hidden_size)
nn_model.train(X_train.values, y_train, iterations, learning_rate)

# K-Nearest Neighbors (KNN) model
knn_model = KNeighborsRegressor(n_neighbors=5)
knn_model.fit(X_train, y_train)

# Test function to predict popularity using both models
def predict_popularity(test_features):
    test_features_normalized = (test_features - X.mean()) / X.std()
    nn_prediction = nn_model.forward(test_features_normalized.values)
    knn_prediction = knn_model.predict(test_features_normalized)
    return nn_prediction, knn_prediction

# Example test cases
test_features1 = pd.DataFrame([[0.8, 0.7, -5.0, 0.1, 0.1, 0.0, 0.1, 0.5, 120]], columns=X.columns)
test_features2 = pd.DataFrame([[0.6, 0.5, -7.0, 0.05, 0.2, 0.0, 0.2, 0.6, 100]], columns=X.columns)

nn_pred1, knn_pred1 = predict_popularity(test_features1)
nn_pred2, knn_pred2 = predict_popularity(test_features2)

print(f"Predicted popularity for test case 1 - Neural Network: {nn_pred1}, KNN: {knn_pred1}")
print(f"Predicted popularity for test case 2 - Neural Network: {nn_pred2}, KNN: {knn_pred2}")


