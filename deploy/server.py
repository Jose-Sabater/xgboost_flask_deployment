from flask import Flask, request, jsonify
from .inference import PredictionModel, MakePrediction
import io
import pandas as pd

app = Flask(__name__)

prediction_model = PredictionModel()
model, scaler = prediction_model.model_and_scaler


@app.route("/predict", methods=["POST"])
def predict_value():
    """Make a prediction based on the data recieved thorugh the post request

    Returns:
        JSON: Prediction
    """
    # Check if the file was provided
    print("request.files:", request.files)
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No file provided"}), 400

    # Read the file content as a pandas DataFrame
    try:
        file_content = file.read()
        data = pd.read_csv(io.StringIO(file_content.decode("utf-8")))
        print(data.head())
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    # Make prediction using model loaded from disk
    prediction = MakePrediction(model, scaler).make_prediction(data)

    # Convert the predictions into a JSON response
    response = {"predictions": prediction.tolist()}
    return jsonify(response)


@app.route("/health")
def health_check():
    return "OK"


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
