import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np

# Load the data
with open('data.pickle', 'rb') as f:
    data = pickle.load(f)

# Access the features and labels
features = data['data']
labels = data['labels']

# Print some basic info
print(f"Number of samples: {len(features)}")
print(f"Number of features per sample: {len(features[0]) if features else 0}")
print(f"Unique classes: {set(labels)}")

# Split data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(
    features, labels, test_size=0.2, shuffle=True, stratify=labels)

# Create and train the model
model = RandomForestClassifier()
model.fit(x_train, y_train)

# Make predictions and evaluate
y_predict = model.predict(x_test)
score = accuracy_score(y_predict, y_test)

print('{}% of samples were classified correctly!'.format(score * 100))

# Save the model
with open('model.p', 'wb') as f:
    pickle.dump({'model': model}, f)

print("Model saved as 'model.p'")
