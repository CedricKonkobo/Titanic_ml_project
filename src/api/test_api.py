import requests
import json

BASE_URL = "http://localhost:5000"

def test_health():
    resp = requests.get(f"{BASE_URL}/health")
    print(f"Health: {resp.json()}")

def test_predict():
    payload = {
        "Pclass": 1,
        "Sex": "female",
        "Age": 25,
        "SibSp": 0,
        "Parch": 0,
        "Fare": 100.0,
        "Embarked": "C",
        "Name": "Doe, Mrs. Jane",
        "Ticket": "PC 12345",
        "Cabin": "C85"
    }
    
    resp = requests.post(f"{BASE_URL}/predict", json=payload)
    print(f"Predict: {resp.json()}")

if __name__ == "__main__":
    test_health()
    test_predict()