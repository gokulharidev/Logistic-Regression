import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

class Logistic:
    def __init__(self, epoch=5000, lr=0.001):
        self.w = None  
        self.b = 0
        self.lr = lr
        self.epoch = epoch

    def sigmoid(self, z):
        z = np.clip(z, -500, 500)  # Prevent overflow
        return 1 / (1 + np.exp(-z))

    def binaryentropy(self, y_pred, y, epsilon=1e-9):
        # Prevent log(0) by clipping predictions
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        return -np.mean(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred))

    def gradient(self, x, y):
        z = np.dot(x, self.w) + self.b
        a = self.sigmoid(z)
        error = y - a
        dw = np.dot(x.T, error)
        db = np.sum(error)
        return a, dw, db

    def fit(self, x, y):
        if self.w is None:
            # Initialize weights with shape (number of features, 1)
            self.w = np.zeros((x.shape[1], 1))
        
        prev_loss = float('inf')
        for i in range(self.epoch):
            a, dw, db = self.gradient(x, y)
            self.w -= self.lr * dw
            self.b -= self.lr * db

            loss = self.binaryentropy(a, y)
            # Early stopping if loss improvement is very small
            if abs(prev_loss - loss) < 1e-6:  
                print(f"Early stopping at epoch {i+1}, Loss: {loss:.6f}")
                break
            prev_loss = loss

        return loss

    def predict(self, x):
        z = np.dot(x, self.w) + self.b
        return self.sigmoid(z)

    def accuracy(self, y, y_pred):
        y_pred_labels = (y_pred >= 0.5).astype(int)
        return np.mean(y == y_pred_labels)

# ---------------------------
# Load and preprocess dataset
# ---------------------------
file = pd.read_csv("X:\\python programs\\ml\\Logistic Regression\\samdataset.csv")

# Assume the first column is the feature and the second column is the target
# Ensure x is a 2D array and y is a column vector
x = file.iloc[:, [0]].values  
y = file.iloc[:, 1].values.reshape(-1, 1)

# Split data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Scale features to have zero mean and unit variance
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# ---------------------------
# Train and evaluate the model
# ---------------------------
model = Logistic(epoch=5000, lr=0.001)
loss = model.fit(x_train, y_train)
print("Final Loss:", loss)

y_pred = model.predict(x_test)
print("Accuracy:", model.accuracy(y_test, y_pred))