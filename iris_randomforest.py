# iris_randomforest.py
from flask import Flask, request, jsonify, render_template
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import pickle

app = Flask(__name__)

# Load the dataset
df = pd.read_csv("./iris.csv")

df = df.drop("Id", axis=1)  # Drop the Id column

print(df.sample(5))  # Display 5 random rows from the dataset
print(df.columns)  # Display the column names

# Separate the features and the target variable
X = df.drop("Species", axis=1)  # Features
y = df["Species"]  # Target variable


# Separate the features and the target variable
# X = df.iloc[:, 0:4].values  # It means all rows and 0 to 3 columns
# y = df.iloc[:, 4].values  # It means all rows and 4th column

# Encode the target variable not required for Random Forest as it can handle categorical data directly without encoding.
# le = LabelEncoder()  # LabelEncoder is used to convert the target variable to integers
# y = le.fit_transform(
#     y
# )  # fit_transform method is used to fit the label encoder and return encoded labels

# Split the dataset into training and testing sets

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Train a Random Forest Classifier on the training data
rfc = RandomForestClassifier(n_estimators=100)  # Create a Random Forest Classifier
rfc.fit(X_train, y_train)  # Train the model

# Accuracy check
y_train_pred = rfc.predict(X_train)
y_test_pred = rfc.predict(X_test)

accuracy_train = np.mean(y_train == y_train_pred)
accuracy_test = np.mean(y_test == y_test_pred)

print("Train accuracy:", accuracy_train)
print("Test accuracy:", accuracy_test)

# Save the trained model using pickle module so that it can be used later for prediction
with open("irirf.pkl", "wb") as f:
    pickle.dump(
        rfc, f
    )  # Save the model to a file named "iri.pkl" in the current working directory

# Load the model
model = pickle.load(open("irirf.pkl", "rb"))


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    features = [
        request.form.get("sepal_length"),
        request.form.get("sepal_width"),
        request.form.get("petal_length"),
        request.form.get("petal_width"),
    ]
    float_features = [float(x) for x in features]
    prediction = model.predict([np.array(float_features)])
    output = prediction[
        0
    ]  # le.inverse_transform(prediction) not required for Random Forest as it can handle categorical data directly without encoding.
    formatted_features = (
        "Sepal Length: {}, Sepal Width: {}, Petal Length: {}, Petal Width: {}".format(
            *features
        )
    )
    return render_template(
        "index.html",
        prediction_text=".. Predicted class for ({}) is {}".format(
            formatted_features, output
        ),
    )


if __name__ == "__main__":
    app.run(port=5000, debug=True)
