import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# Define the logistic function (sigmoid function)
def logistic_function(z):
    return 1 / (1 + np.exp(-z))

# Define the Binary Cross-Entropy Loss
def binary_cross_entropy_loss(y_true, y_pred):
    epsilon = 1e-15  # To prevent log(0) issues
    loss = - (y_true * np.log(y_pred + epsilon) + (1 - y_true) * np.log(1 - y_pred + epsilon))
    return np.mean(loss)

whitew = pd.read_csv("winequality-white.csv", sep=';')
redw = pd.read_csv("winequality-red.csv", sep=';')
whitew = whitew.head(1600)
whitew['wine_type'] = 0  # 0 for white wine
redw['wine_type'] = 1  # 1 for red wine

# Combine the DataFrames while shuffling the rows
wine_data = pd.concat([whitew, redw], ignore_index=True).sample(frac=1, random_state=42)

# Reset the index of the combined DataFrame
wine_data.reset_index(drop=True, inplace=True)

selected_features = ["fixed acidity","volatile acidity","citric acid","residual sugar","chlorides","free sulfur dioxide","total sulfur dioxide","density","pH","sulphates","alcohol"]
X = wine_data[selected_features]
y = wine_data['wine_type']  # Target variable

# Split the data into training and testing sets for Logistic Regression
X_train_lr, X_test_lr, y_train_lr, y_test_lr = train_test_split(X, y, test_size=0.1, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train_lr = scaler.fit_transform(X_train_lr)
X_test_lr = scaler.transform(X_test_lr)

# Initialize model parameters for Logistic Regression
learning_rate_lr = 0.01
num_epochs_lr = 1000
theta_lr = np.zeros(X_train_lr.shape[1])
train_loss_history_lr = []

# Training loop for Logistic Regression
for epoch in range(num_epochs_lr):
    z = np.dot(X_train_lr, theta_lr)
    predictions = logistic_function(z)
    
    # Calculate the Binary Cross-Entropy Loss for Logistic Regression
    loss = binary_cross_entropy_loss(y_train_lr, predictions)
    train_loss_history_lr.append(loss)
    
    # Calculate the gradient of the loss with respect to theta for Logistic Regression
    gradient = np.dot(X_train_lr.T, (predictions - y_train_lr)) / y_train_lr.size
    
    # Update model parameters using gradient descent for Logistic Regression
    theta_lr -= learning_rate_lr * gradient

# Make predictions on the test set for Logistic Regression
z_test_lr = np.dot(X_test_lr, theta_lr)
y_pred_lr = logistic_function(z_test_lr)

# Threshold predictions to get binary labels (0 or 1) for Logistic Regression
y_pred_binary_lr = np.round(y_pred_lr)

# Calculate accuracy for Logistic Regression
accuracy_lr = accuracy_score(y_test_lr, y_pred_binary_lr)

# Print the accuracy for Logistic Regression
print("Logistic Regression Accuracy: {:.2f}".format(accuracy_lr))

# Plot the training loss over epochs for Logistic Regression
plt.figure(figsize=(6, 6))
plt.plot(range(num_epochs_lr), train_loss_history_lr)
plt.xlabel('Epochs')
plt.ylabel('Training Loss (Logistic Regression)')
plt.title('Training Loss vs. Epochs (Logistic Regression)')
plt.show()
