import pandas as pd
import pickle
from sklearn.neighbors import KNeighborsClassifier

# Load dataset
df = pd.read_csv('dataset.csv')

# Define features and target
features = ['weight', 'age', 'height']
target = 'size'

X_train = df[features]
y_train = df[target]

# Build and train model
k_value = 7
model = KNeighborsClassifier(n_neighbors=k_value)
model.fit(X_train, y_train)

# Save model
with open('model.pkl', 'wb') as f:
    pickle.dump((model, y_train, k_value), f)