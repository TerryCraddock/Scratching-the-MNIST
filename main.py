import numpy as np  # for numerical operations
import pandas as pd  # for data manipulation
from matplotlib import pyplot as plt  # for plotting (not used in this script)

# Load the MNIST dataset. The dataset is assumed to be in csv format.
# Each row of the dataset represents an image: the first element of each row is the digit label and the rest 784 elements are pixel values.
data = pd.read_csv('Data/mnist_train.csv')
data = np.array(data)
m, n = data.shape

# Shuffle the data. This is important in machine learning to ensure that the model gets to see all variations of the data and enhance the model's generalization ability.
np.random.shuffle(data)

# Split the data into development (or test) and training sets (80-20 split).
num_dev = int(m * 0.2)  # 20% of data for development set
data_dev = data[:num_dev]  # first 20% data rows
data_train = data[num_dev:]  # next 80% data rows

# Separate the labels (Y) and the features (X). Also, normalize the pixel values by dividing by 255.0.
Y_dev = data_dev[:, 0]
X_dev = data_dev[:, 1:] / 255.0  # normalize pixel values to range [0, 1]
Y_train = data_train[:, 0]
X_train = data_train[:, 1:] / 255.0  # normalize pixel values to range [0, 1]

def init_params():
    """Initialize the weights and biases for the network."""
    # Use He initialization (multiply the random values with sqrt(2/n)) where n is the number of units in the previous layer.
    W1 = np.random.randn(10, 784) * np.sqrt(2./784)
    b1 = np.zeros((10, 1))  # biases for the first layer is set to zeros
    W2 = np.random.randn(10, 10) * np.sqrt(2./10)
    b2 = np.zeros((10, 1))  # biases for the second layer is set to zeros
    return W1, b1, W2, b2

def ReLU(Z):
    """ReLU activation function."""
    return np.maximum(0, Z)  # returns Z if Z > 0, else returns 0

def softmax(Z):
    """Softmax activation function."""
    # Subtract max(Z) to make the function numerically stable.
    Z_exp = np.exp(Z - np.max(Z, axis=0))
    return Z_exp / np.sum(Z_exp, axis=0)

def forward_prop(W1, b1, W2, b2, X):
    """Perform a forward pass through the network."""
    # First layer computations
    Z1 = W1.dot(X.T) + b1  # Transpose X for matrix multiplication
    A1 = ReLU(Z1)  # Apply ReLU activation function

    # Second layer (output layer) computations
    Z2 = W2.dot(A1) + b2
    A2 = softmax(Z2)  # Apply softmax activation function

    return Z1, A1, Z2, A2

def one_hot(Y):
    """One-hot encode labels"""
    one_hot_Y = np.zeros((Y.size, Y.max() + 1))
    one_hot_Y[np.arange(Y.size), Y] = 1
    one_hot_Y = one_hot_Y.T
    return one_hot_Y

def deriv_ReLU(Z):
    """Derivative of the ReLU function."""
    return Z > 0  # derivative of ReLU function is 1 for x > 0, else 0

def back_prop(Z1, A1, Z2, A2, W2, X, Y):
    """Calculate derivatives using backpropagation."""
    m = Y.size
    one_hot_y = one_hot(Y)  # Get one-hot encoded labels
    dZ2 = A2 - one_hot_y
    dW2 = 1 / m * dZ2.dot(A1.T)
    db2 = 1 / m * np.sum(dZ2, axis=1, keepdims=True)
    dZ1 = W2.T.dot(dZ2) * deriv_ReLU(Z1)
    dW1 = 1 / m * dZ1.dot(X)
    db1 = 1 / m * np.sum(dZ1, axis=1, keepdims=True)
    return dW1, db1, dW2, db2

def update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha):
    """Update the parameters using the gradients computed in backpropagation."""
    W1 = W1 - alpha * dW1  # update weights for first layer
    b1 = b1 - alpha * db1  # update biases for first layer
    W2 = W2 - alpha * dW2  # update weights for second layer
    b2 = b2 - alpha * db2  # update biases for second layer
    return W1, b1, W2, b2

def get_predictions(A2):
    """Return the predictions by taking the class with maximum probability in the output layer."""
    return np.argmax(A2, axis=0)

def get_accuracy(predictions, Y):
    """Calculate the accuracy of predictions against the true values."""
    return np.sum(predictions == Y) / Y.size

def gradient_descent(X, Y, iterations, alpha):
    """Perform the full training by using gradient descent."""
    W1, b1, W2, b2 = init_params()
    for i in range(iterations):
        Z1, A1, Z2, A2 = forward_prop(W1, b1, W2, b2, X)
        dW1, db1, dW2, db2 = back_prop(Z1, A1, Z2, A2, W2, X, Y)
        W1, b1, W2, b2 = update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha)
        if i % 10 == 0:
            print("Iteration: ", i)
            predictions = get_predictions(A2)
            print("Accuracy: ", get_accuracy(predictions, Y))
    return W1, b1, W2, b2

def main():
    """Main function to tie all the logic together."""
    W1, b1, W2, b2 = gradient_descent(X_train, Y_train, 100, 0.25)

if __name__ == "__main__":
    main()
