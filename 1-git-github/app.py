from flask import Flask, request, jsonify
import pickle
import pandas as pd

app = Flask(__name__)

# Load the model
with open('model.pkl', 'rb') as f:
    model, y_train, k_value = pickle.load(f)


# Route for prediction
@app.route('/predict', methods=['POST'])
def predict():
    # Receive JSON request
    data = request.get_json(force=True)
    user_input = [
        data['weight'],
        data['age'],
        data['height'],
    ]

    # Perform prediction
    distances, neighbors = model.kneighbors([user_input], n_neighbors=2*k_value, return_distance=True)

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

    # Return predicition result as JSON
    return jsonify({'prediction': most_common_label})

if __name__ == '__main__':
    app.run(port=5000, debug=True)