import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns

# Load the wine dataset (assuming you have "winequality-red.csv" and "winequality-white.csv" files)
red_wine = pd.read_csv("winequality-red.csv", sep=';')
white_wine = pd.read_csv("winequality-white.csv", sep=';')

# Add a 'wine_type' column (0 for white, 1 for red)
red_wine['wine_type'] = 1
white_wine['wine_type'] = 0

# Combine the two datasets
wine_data = pd.concat([red_wine, white_wine], ignore_index=True)

# Select features and target variable
selected_features = ["fixed acidity","volatile acidity","citric acid","residual sugar","chlorides","free sulfur dioxide","total sulfur dioxide","density","pH","sulphates","alcohol"]
X = wine_data[selected_features]
y = wine_data['wine_type']

# Split the data into a training set and a testing set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Split the remaining data into a validation set and a test set
X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=0.5, random_state=42)

# Create a Decision Tree classifier
decision_tree = DecisionTreeClassifier(random_state=42)

# Train the model on the training data
decision_tree.fit(X_train, y_train)

# Make predictions on the test data
y_pred = decision_tree.predict(X_test)

# Evaluate the model's performance on the test set
accuracy = accuracy_score(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

# Print the results for the test set
print("Results on Test Set:")
print(f"Accuracy: {accuracy:.2f}")
print("Classification Report:\n", classification_rep)
print("Confusion Matrix:\n", conf_matrix)

# Make predictions on the validation data
y_val_pred = decision_tree.predict(X_val)

# Calculate the Mean Squared Error (MSE) for the validation set
mse_val = mean_squared_error(y_val, y_val_pred)

# Print the MSE for the validation set
print(f"Mean Squared Error (MSE) on Validation Set: {mse_val:.2f}")

# Visualize the Decision Tree structure
plt.figure(figsize=(12, 6))
plot_tree(decision_tree, feature_names=selected_features, class_names=["White", "Red"], filled=True, rounded=True)
plt.title("Decision Tree Structure")
plt.show()

# Visualize feature importances
feature_importances = decision_tree.feature_importances_
plt.figure(figsize=(10, 6))
sns.barplot(x=feature_importances, y=selected_features)
plt.title("Feature Importances")
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.show()
