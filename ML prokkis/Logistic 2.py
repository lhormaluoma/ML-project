import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import seaborn as sns

whitew=pd.read_csv("winequality-white.csv", sep=';')
redw=pd.read_csv("winequality-white.csv", sep=';')
whitew=whitew.head(1600)
whitew['wine_type'] = 0  # 0 for white wine
redw['wine_type'] = 1  # 1 for red wine

# Combine the DataFrames while shuffling the rows
wine_data = pd.concat([whitew, redw], ignore_index=True).sample(frac=1, random_state=42)

# Reset the index of the combined DataFrame
wine_data.reset_index(drop=True, inplace=True)

selected_features = ['fixed acidity', 'volatile acidity', 'citric acid', 'alcohol']
X = wine_data[selected_features]
y = wine_data['wine_type']  # Target variable

# Split the data into a training set and a testing set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Create a logistic regression model
model = LogisticRegression(random_state=42)

# Train the model on the training data
model.fit(X_train, y_train)

# Make predictions on the test data
y_pred = model.predict(X_test)

# Evaluate the model's performance
accuracy = accuracy_score(y_test, y_pred)

# Generate a classification report without averaging
classification_rep = classification_report(y_test, y_pred, zero_division=0)

# Create a confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)

# Print the results
print(f"Accuracy: {accuracy:.2f}")
print("Classification Report:\n", classification_rep)
print("Confusion Matrix:\n", conf_matrix)

# Visualize the results
plt.figure(figsize=(10, 6))

# Plot the confusion matrix as a heatmap
plt.subplot(1, 2, 1)
sns.heatmap(conf_matrix, annot=True, cmap="Blues", fmt="d", cbar=False)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')

# Plot precision and recall for each class
plt.subplot(1, 2, 2)
report_df = pd.DataFrame.from_dict(classification_report(y_test, y_pred, output_dict=True))
precision_recall = report_df[['0', '1']]  # Use class labels '0' and '1'
precision_recall = precision_recall.T
sns.heatmap(precision_recall, annot=True, cmap="YlGnBu")
plt.title('Precision & Recall')

plt.tight_layout()
plt.show()
