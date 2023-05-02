import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.neighbors import KNeighborsRegressor
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

files = [
    "webscraping/Billboard Hot 100.csv",
    "webscraping/BUTTER.csv",
    "webscraping/Electric Lady Studios.csv",
    "webscraping/Fresh Finds.csv",
    "webscraping/Best of the Decade For You.csv",
]
frames = []
for f in files:
    frames.append(pd.read_csv(f, encoding="cp1252"))
df = pd.concat(frames)

# Remove leading and trailing spaces from column names
df.columns = [col.strip() for col in df.columns]

# Normalize the features
X = df.iloc[:, 1:-1]
X_normalized = (X - X.mean()) / X.std()
y = df["popularity"]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_normalized, y, test_size=0.2, random_state=42)

# Neural network model
nn_model = Sequential()  # Initialize the neural network model

# Add input layer with 32 nodes and ReLU activation function
nn_model.add(Dense(32, activation="relu", input_shape=(X_train.shape[1],)))

# Add hidden layer with 16 nodes and ReLU activation function
nn_model.add(Dense(16, activation="relu"))

# Add output layer with a single node for regression task
nn_model.add(Dense(1))

# Compile the neural network model using the Adam optimizer and Mean Squared Error as the loss function
nn_model.compile(optimizer="adam", loss="mean_squared_error")

# Train the neural network model
# Epochs: 100 (number of times the entire dataset is passed through the network)
# Batch size: 32 (number of samples per gradient update)
nn_model.fit(X_train, y_train, epochs=100, batch_size=32, verbose=0)

# K-Nearest Neighbors (KNN) model
knn_model = KNeighborsRegressor(n_neighbors=5)  # Initialize the KNN model with 5 neighbors
knn_model.fit(X_train, y_train)  # Train the KNN model using the training data

# Evaluate the models
nn_pred = nn_model.predict(X_test)  # Predict popularity with the Neural Network model on the test set
knn_pred = knn_model.predict(X_test)  # Predict popularity with the KNN model on the test set

nn_mse = mean_squared_error(y_test, nn_pred)  # Calculate the Mean Squared Error for the Neural Network model
knn_mse = mean_squared_error(y_test, knn_pred)  # Calculate the Mean Squared Error for the KNN model

print(f"Neural Network MSE: {nn_mse}")
print(f"K-Nearest Neighbors MSE: {knn_mse}")

def predict_popularity(danceability, energy, loudness, speechiness, acousticness, instrumentalness, liveness, valence, tempo):
    # Create a new DataFrame with the input features
    input_data = pd.DataFrame(
        {
            "danceability": [danceability],
            "energy": [energy],
            "loudness": [loudness],
            "speechiness": [speechiness],
            "acousticness": [acousticness],
            "instrumentalness": [instrumentalness],
            "liveness": [liveness],
            "valence": [valence],
            "tempo": [tempo],
        }
    )

    # Normalize the input features
    input_normalized = (input_data - X.mean()) / X.std()

    # Predict popularity using both models
    nn_pred = nn_model.predict(input_normalized)
    knn_pred = knn_model.predict(input_normalized)

    return nn_pred[0][0], knn_pred[0]

def process_examples(examples):
    results = []  # Initialize an empty list to store the results
    
    # Iterate through the examples in the input list
    for example in examples:
        # Unpack the features from the example tuple
        danceability, energy, loudness, speechiness, acousticness, instrumentalness, liveness, valence, tempo = example
        
        # Call the predict_popularity() function to get the predicted popularity for both models
        nn_popularity, knn_popularity = predict_popularity(danceability, energy, loudness, speechiness, acousticness, instrumentalness, liveness, valence, tempo)
        
        # Create a dictionary with the input example and the predicted popularity values
        result = {
            "input": example,
            "nn_popularity": nn_popularity,
            "knn_popularity": knn_popularity,
        }
        
        # Append the dictionary to the results list
        results.append(result)
    
    # Return the list of dictionaries containing the input examples and the predicted popularity values
    return results

# Test the function with a list of examples
test_examples = [
    (0.7, 0.6, -6.0, 0.05, 0.1, 0.0001, 0.12, 0.4, 120.0),
    (0.8, 0.5, -5.0, 0.04, 0.2, 0.0002, 0.10, 0.3, 110.0),
]

# Call the process_examples() function with the test_examples list and store the returned results in the predictions variable
predictions = process_examples(test_examples)

# Iterate through the predictions and print the input example and the predicted popularity values for both models
for i, prediction in enumerate(predictions):
    print(f"Example {i + 1}:")
    print(f"  Input: {prediction['input']}")
    print(f"  Neural Network predicted popularity: {prediction['nn_popularity']}")
    print(f"  K-Nearest Neighbors predicted popularity: {prediction['knn_popularity']}")
    print()














