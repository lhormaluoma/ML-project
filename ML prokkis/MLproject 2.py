import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

whitew = pd.read_csv("winequality-white.csv", sep=';')
redw = pd.read_csv("winequality-white.csv", sep=';')

whitew['wine_type'] = 1  # 0 for white wine
redw['wine_type'] = 0  # 1 for red wine

# Combine the DataFrames while shuffling the rows
wine_data = pd.concat([whitew, redw], ignore_index=True).sample(frac=1, random_state=42)
# Split the data into features (X) and target (y)
X = wine_data.iloc[:, 0:4]
y = wine_data['wine_type']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=5)

# Feature scaling (Standardization)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Try different classification algorithms
models = {
    'Random Forest': RandomForestClassifier(random_state=42),
    'Support Vector Classifier': SVC(random_state=42)
}

for model_name, model in models.items():
    # Cross-validation to assess model performance
    cv_scores = cross_val_score(model, X_train, y_train, cv=5)
    print(f"Cross-validation scores for {model_name}:", cv_scores)
    print(f"Mean accuracy for {model_name}: {cv_scores.mean():.2f}\n")

    # Train the model on the entire training set
    model.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = model.predict(X_test)

    # Swap the predictions (0s become 1s and vice versa)
    y_pred_swapped = 1 - y_pred

    # Evaluate accuracy with swapped predictions
    accuracy = accuracy_score(y_test, y_pred_swapped)
    print(f"Accuracy for {model_name}: {accuracy:.2f}\n")

    # Generate a classification report as a string
    classification_rep_str = classification_report(y_test, y_pred_swapped)
    print(f"Classification Report for {model_name}:\n{classification_rep_str}\n")

    # Create a confusion matrix
    conf_matrix = confusion_matrix(y_test, y_pred_swapped)


    # Visualize the results
    plt.figure(figsize=(12, 6))

    # Plot the confusion matrix as a heatmap
    plt.subplot(1, 2, 1)
    sns.heatmap(conf_matrix, annot=True, cmap="Blues", fmt="d", cbar=False)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(f'Confusion Matrix - {model_name}')

    plt.tight_layout()
    plt.show()
