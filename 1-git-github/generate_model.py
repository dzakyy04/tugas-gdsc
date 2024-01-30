import pandas as pd
import pickle
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from tabulate import tabulate

# Load dataset
df = pd.read_csv('data_training.csv')

# Define features and target
features = ['weight', 'age', 'height']
target = 'size'

X_train = df[features]
y_train = df[target]

# Build and train model
k_value = 6
model = KNeighborsClassifier(n_neighbors=k_value)
model.fit(X_train, y_train)

# Save model
with open('model.pkl', 'wb') as f:
    pickle.dump((model, y_train, k_value), f)

# Load model
with open('model.pkl', 'rb') as f:
    model, y_train, k_value = pickle.load(f)

# Read dataset testing
df_test = pd.read_csv('data_testing.csv')
X_test = df_test[features]
y_test = df_test[target]

# Perform prediction for testing data
y_pred = model.predict(X_test)

# Generate confusion matrix
labels_order = ['S', 'M', 'L', 'XL', 'XXL']
cm = confusion_matrix(y_test, y_pred, labels=labels_order)
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt='', cmap='Blues', xticklabels=labels_order, yticklabels=labels_order).set(xlabel='Prediction', ylabel='Actual')
plt.show()

# Calculate accuracy, precision, recall, and f1-score for each class
precision = precision_score(y_test, y_pred, average=None, labels=labels_order)
recall = recall_score(y_test, y_pred, average=None, labels=labels_order)
f1 = f1_score(y_test, y_pred, average=None, labels=labels_order)

metrics_table = []
for i, label in enumerate(labels_order):
    metrics_table.append([label, precision[i], recall[i], f1[i]])

print(tabulate(metrics_table, headers=["Class", "Precision", "Recall", "F1 Score"], tablefmt="pretty"))

# Calculate accuracy, precision, recall, and f1-score
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

print("\nOverall:")
print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 Score: {f1}")