# DO NOT CHANGE THE NAME OF THIS METHOD OR ITS INPUT OUTPUT BEHAVIOR

# INPUT CONVENTION
# filenames: a list of strings containing filenames of images

# OUTPUT CONVENTION
# The method must return a list of strings. Make sure each string is either "ODD"
# or "EVEN" (without the quotes) depending on whether the hexadecimal number in
# the image is odd or even. Take care not to make spelling or case mistakes. Make
# sure that the length of the list returned as output is the same as the number of
# filenames that were given as input. The judge may give unexpected results if this
# convention is not followed.

import cv2
import numpy as np
import os
from sklearn.linear_model import LogisticRegression

# Load training data and labels
train_data = []
labels = []
i = 0
with open("/content/gdrive/MyDrive/assn2/train/labels.txt", "r") as file:
    lines = file.readlines()
    for line in lines:
        label = line.strip().split()[0]  # Only use the first character of the label
        image = cv2.imread(f"/content/gdrive/MyDrive/assn2/train/{i}.png", 0)  # Read image in grayscale
        train_data.append(image)
        labels.append(label)
        i += 1

# Preprocess training data
preprocessed_data = []
for image in train_data:
    # Apply preprocessing techniques here (e.g., normalization, resizing, etc.)
    # Modify the code based on your chosen techniques
    preprocessed_image = image
    preprocessed_data.append(preprocessed_image)

# Convert data and labels to numpy arrays
X_train = np.array(preprocessed_data)
y_train = np.array(labels)

# Convert labels to integer labels
label_mapping = {'ODD': 0, 'EVEN': 1}
y_train = np.array([label_mapping[label] for label in y_train])

# Flatten the data for logistic regression
X_train_flat = X_train.reshape(X_train.shape[0], -1)

# Train the logistic regression model
model = LogisticRegression()
model.fit(X_train_flat, y_train)

# Define the decaptcha function
def decaptcha(filenames):
    predicted_labels = []
    for filename in filenames:
        # Read image in grayscale
        image = cv2.imread(filename, 0)
        # Preprocess the image (apply necessary transformations)
        preprocessed_image = image
        # Flatten the preprocessed image for prediction
        flattened_image = preprocessed_image.reshape(1, -1)
        # Make prediction using the logistic regression model
        prediction = model.predict(flattened_image)
        # Map the prediction to the label ("ODD" or "EVEN")
        label = "ODD" if prediction == 0 else "EVEN"
        # Append the label to the list of predicted labels
        predicted_labels.append(label)
    # Return the list of predicted labels
    return predicted_labels
