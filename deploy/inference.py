import pandas as pd
import numpy as np
import joblib

import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


class PredictionModel:
    """Singleton class to load the model and scaler"""

    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(
        self, model_path="./prod_model/model.pkl", scaler_path="./prod_model/scaler.pkl"
    ):
        if self._initialized:
            return
        self.model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)
        self._initialized = True

    @property
    def model_and_scaler(self):
        return self.model, self.scaler


class MakePrediction:
    """Class to make predictions"""

    def __init__(self, model, scaler):
        self.model = model
        self.scaler = scaler

    def make_prediction(self, data: pd.DataFrame, scale=True) -> pd.DataFrame:
        """Make a prediction based on the data

        Args:
            data (pd.DataFrame): Data to be used for prediction
            scale (bool, optional): Whether to scale the data. Defaults to True.

            Returns:
                pd.DataFrame: Prediction

        """
        raw = pd.DataFrame(data)
        df = self.data_preparation(raw)
        X = self.define_X(df)
        if scale:
            X = self.scaler.transform(X)
        prediction = self.model.predict(X)
        return prediction

    @staticmethod
    def data_preparation(data: pd.DataFrame) -> pd.DataFrame:
        """Clean data by removing outliers and missing values and creating new features

        Args:
            data (pd.DataFrame): Data to be cleaned

        Returns:
            pd.DataFrame: Cleaned data ready for inference
        """

        df = data.copy()
        df = df[df["passenger_count"] > 0]
        df = df[df["trip_distance"] > 0]
        df = df[df["fare_amount"] > 0]
        df = df[df["total_amount"] > 0]

        df["tpep_pickup_datetime"] = pd.to_datetime(df["tpep_pickup_datetime"])
        df["tpep_dropoff_datetime"] = pd.to_datetime(df["tpep_dropoff_datetime"])

        # create filtering features
        df = df.assign(day_of_week=df["tpep_pickup_datetime"].dt.dayofweek)
        df = df.assign(weekday=df["tpep_pickup_datetime"].dt.day_name())
        df = df.assign(weeknr=df["tpep_pickup_datetime"].dt.isocalendar().week)
        df = df.assign(hour_of_day=df["tpep_pickup_datetime"].dt.hour)
        df = df.assign(month=df["tpep_pickup_datetime"].dt.month)
        df = df.assign(year=df["tpep_pickup_datetime"].dt.year)
        df = df.sort_values(by="tpep_pickup_datetime")

        # remove missing values if they are in the columns we need
        df = df.dropna(
            subset=[
                "trip_distance",
                "passenger_count",
                "PULocationID",
                "DOLocationID",
                "RatecodeID",
                "hour_of_day",
                "day_of_week",
                "weeknr",
            ]
        )
        print(
            f"Nr of values removed due to missing values: {data.shape[0] - df.shape[0]}"
        )
        return df.reset_index()

    @staticmethod
    def define_X(df: pd.DataFrame) -> np.ndarray:
        """Define X for the model based on columns: "trip_distance",
            "passenger_count",
            "PULocationID",
            "DOLocationID",
            "RatecodeID",
            "hour_of_day",
            "day_of_week",
            "weeknr"

        Args:
            df (pd.DataFrame): Data to be used

        Returns:
            np.ndarray: X and y
        """

        X = df[
            [
                "trip_distance",
                "passenger_count",
                "PULocationID",
                "DOLocationID",
                "RatecodeID",
                "hour_of_day",
                "day_of_week",
                "weeknr",
            ]
        ].values

        return X
