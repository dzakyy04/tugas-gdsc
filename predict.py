import pickle
import pandas as pd

# Load the model
with open('model.pkl', 'rb') as f:
    model, y_train, k_value = pickle.load(f)

# Receive input from the user
weight = float(input("Enter the weight: "))
age = float(input("Enter the age: "))
height = float(input("Enter the height: "))

# Convert the input to a DataFrame with feature names
user_input_df = pd.DataFrame([[weight, age, height]], columns=['weight', 'age', 'height'])

# Perform prediction
distances, neighbors = model.kneighbors(user_input_df, n_neighbors=2*k_value, return_distance=True)

nearest_neighbors = []
k_count = 0
for i, distance in enumerate(distances[0]):
    if i > 0 and distance == distances[0][i-1]:
        k_count -= 1
    neighbor_info = {
        'Label': y_train.iloc[neighbors[0][i]],
        'Distance': distance,
        'K': k_count + 1
    }
    nearest_neighbors.append(neighbor_info)
    k_count += 1
    if k_count == k_value:
        break

labels = [neighbor['Label'] for neighbor in nearest_neighbors]
most_common_label = pd.Series(labels).mode()[0]

print(f"\nPrediction: {most_common_label}")
