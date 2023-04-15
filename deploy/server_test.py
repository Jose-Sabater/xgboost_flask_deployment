import requests
import pandas as pd

print("starting test")
url = "http://127.0.0.1:5000/predict"
# url = "http://192.168.0.18:5000"

file_path = "./test/test.csv"

with open(file_path, "rb") as file:
    print("file opened")
    files = {"file": (file_path, file)}
    response = requests.post(url, files=files)
    print(f"response: {response}")

if response.status_code == 200:
    print("Predictions:", response.json())
    print("Last 5 predictions:", response.json()["predictions"][-5:])
else:
    print("Error:", response.json(), "Status Code:", response.status_code)
