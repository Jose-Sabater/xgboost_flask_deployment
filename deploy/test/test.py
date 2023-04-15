import pandas as pd
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from inference import PredictionModel, MakePrediction


def test_multiple_rows_prediction():
    prediction_model = PredictionModel()

    model, scaler = prediction_model.model_and_scaler

    data = pd.read_csv("./test/test.csv")
    prediction = MakePrediction(model, scaler).make_prediction(data)

    assert (
        prediction.shape[0] == 290
    ), "Incorrect number of predictions for the multiple rows test"


def test_single_row_prediction():
    prediction_model = PredictionModel()
    model, scaler = prediction_model.model_and_scaler

    # read single row
    data = pd.read_csv("./test/test.csv", nrows=1)
    print(data)
    prediction = MakePrediction(model, scaler).make_prediction(data)

    assert (
        prediction.shape[0] == 1
    ), "Incorrect number of predictions for the single row test"
