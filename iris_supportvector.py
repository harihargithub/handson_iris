# iris.py
from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
import pickle

app = Flask(__name__)


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(force=True)
    prediction = model.predict([np.array(data["features"])])
    output = le.inverse_transform(prediction)
    return jsonify(output[0])


if __name__ == "__main__":
    # Load the dataset
    df = pd.read_csv("./iris.csv")

    # Separate the features and the target variable
    X = df.iloc[:, 0:4].values  # It means all rows and 0 to 3 columns
    y = df.iloc[:, 4].values  # It means all rows and 4th column

    # Encode the target variable
    le = (
        LabelEncoder()
    )  # LabelEncoder is used to convert the target variable to integers
    y = le.fit_transform(
        y
    )  # fit_transform method is used to fit the label encoder and return encoded labels

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Train a Support Vector Classifier on the training data
    svc = SVC(kernel="linear")  # Create a Support Vector Classifier
    svc.fit(X_train, y_train)  # Train the model

    # Save the trained model using pickle module so that it can be used later for prediction
    with open("iri.pkl", "wb") as f:
        pickle.dump(
            svc, f
        )  # Save the model to a file named "iri.pkl" in the current working directory

    # Load the model
    with open("irisv.pkl", "rb") as f:
        model = pickle.load(f)

    app.run(port=5000, debug=True)
