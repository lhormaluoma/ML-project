import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, mean_squared_error, log_loss
import matplotlib.pyplot as plt
import seaborn as sns

# Load the wine dataset (assuming you have "winequality-red.csv" and "winequality-white.csv" files)
whitew = pd.read_csv("winequality-white.csv", sep=';')
redw = pd.read_csv("winequality-red.csv", sep=';')  # Changed to "winequality-red.csv" for red wine data
whitew = whitew.head(1600)
whitew['wine_type'] = 0  # 0 for white wine
redw['wine_type'] = 1  # 1 for red wine

# Combine the two datasets
wine_data = pd.concat([redw, whitew], ignore_index=True)

# Select features and target variable
selected_features = ["volatile acidity","chlorides","total sulfur dioxide","density","sulphates","alcohol"]
X = wine_data[selected_features]
y = wine_data['wine_type']

# Split the data into a training set and a testing set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Split the remaining data into a validation set and a test set
X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=0.5, random_state=42)

# Create used models
model1 = DecisionTreeClassifier(random_state=42)
model2 = LogisticRegression(random_state=42, max_iter=1000)

models = [model1, model2]


for i in range(len(models)):
    models[i].fit(X_train, y_train)

    y_pred = models[i].predict(X_train)

    accuracy = accuracy_score(y_train, y_pred)
    classification_rep = classification_report(y_train, y_pred)
    conf_matrix = confusion_matrix(y_train, y_pred)

    print("Results on Test Set" + str(i) + " :")
    print(f"Accuracy: {accuracy:.2f}")
    print("Classification Report:\n", classification_rep)
    print("Confusion Matrix:\n", conf_matrix)

    # Plot the confusion matrix as a heatmap
    plt.subplot(1, 2, 1)
    sns.heatmap(conf_matrix, annot=True, cmap="Blues", fmt="d", cbar=False)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')

    # Plot precision and recall for each class
    plt.subplot(1, 2, 2)
    report_df = pd.DataFrame(classification_report(y_train, y_pred, output_dict=True))
    precision_recall = report_df[['0', '1']]  # Use class labels '0' and '1'
    precision_recall = precision_recall.T
    sns.heatmap(precision_recall, annot=True, cmap="YlGnBu")
    plt.title('Precision & Recall')

    plt.tight_layout()
    plt.show()

# Calculate the Mean Squared Error (MSE) on the training set, validation set and test set
y_val_train = models[0].predict(X_train)
mse_train = mean_squared_error(y_train, y_val_train)
print(f"Mean Squared Error (MSE) on Training Set: {mse_train:.2f}")

y_val_pred = models[0].predict(X_val)
mse_val = mean_squared_error(y_val, y_val_pred)
print(f"Mean Squared Error (MSE) on Validation Set: {mse_val:.2f}")

y_test_pred = models[0].predict(X_test)
mse_test = mean_squared_error(y_test, y_test_pred)
print(f"Mean Squared Error (MSE) on Test Set: {mse_test:.2f}")

# Visualize the Decision Tree structure
plt.figure(figsize=(12, 6))
plot_tree(models[0], feature_names=selected_features, class_names=["White", "Red"], filled=True, rounded=True)
plt.title("Decision Tree Structure")
plt.show()

# Visualize feature importances
feature_importances = models[0].feature_importances_
plt.figure(figsize=(10, 6))
sns.barplot(x=feature_importances, y=selected_features)
plt.title("Feature Importances")
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.show()

model2.fit(X_train, y_train)
y_pred_train_2 = model2.predict_proba(X_train)

tr_error = log_loss(y_train, y_pred_train_2)
print("Logistic Regression training loss: ", tr_error)

y_pred2 = model2.predict_proba(X_val)
val_error2 = log_loss(y_val, y_pred2)
print("Logistic Regression validation loss: ", val_error2)

model2.fit(X_test, y_test)
y_pred_test_2 = model2.predict_proba(X_test)
test_errs_2 = log_loss(y_test, y_pred_test_2)
print("Logistic Regression test loss :", test_errs_2)