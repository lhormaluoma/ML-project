import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import seaborn as sns

# Load and preprocess the data
whitew = pd.read_csv("winequality-white.csv", sep=';')
redw = pd.read_csv("winequality-red.csv", sep=';')

whitew['wine_type'] = 0  # 0 for white wine
redw['wine_type'] = 1  # 1 for red wine

# Combine the DataFrames while shuffling the rows
wine_data = pd.concat([whitew, redw], ignore_index=True).sample(frac=1, random_state=42)

# Reset the index of the combined DataFrame
wine_data.reset_index(drop=True, inplace=True)

# Select the first 4 features (columns 1-4) for X
X = wine_data.iloc[:, 0:4]  # Features (columns 1-4)

# Target variable (wine_type, column 12)
y = wine_data['wine_type']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

# Create a logistic regression model
model = LogisticRegression(random_state=42)
model.fit(X_train, y_train)

# Make predictions on the test data
y_pred = model.predict(X_test)

# Evaluate accuracy
accuracy = accuracy_score(y_test, y_pred)

# Generate a classification report
classification_rep = classification_report(y_test, y_pred, output_dict=True)

# Create a confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)

# Visualize the results
plt.figure(figsize=(12, 6))

# Plot the confusion matrix as a heatmap
plt.subplot(1, 2, 1)
sns.heatmap(conf_matrix, annot=True, cmap="Blues", fmt="d", cbar=False)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')

# Plot precision and recall for each class
plt.subplot(1, 2, 2)
report_df = pd.DataFrame(classification_rep).iloc[:-1, :-1]  # Exclude the last row and last column
sns.heatmap(report_df, annot=True, cmap="YlGnBu")
plt.title('Precision & Recall')

plt.tight_layout()
plt.show()
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")

# Generate a classification report
classification_rep = classification_report(y_test, y_pred)
print("Classification Report:\n", classification_rep)

# Create a confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", conf_matrix)