import os
import io
import json
import pandas as pd
import numpy as np
from xgboost import Booster


import pickle as pkl

import xgboost as xgb
import tempfile

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from io import StringIO

from flask import Flask, request, jsonify, Response



# Initialize Flask app
app = Flask(__name__)
model = None  # Global model variable

def model_fn(model_dir):
    """
    Load the XGBoost model from the model_dir.
    """
    #model_path = os.path.join(model_dir, "xgboost-model")
    #booster = xgb.Booster()
    #model = Booster()
    #model.load_model(model_path)
    #return model

    print(f"Trying to load model from: {model_dir}")
    model_path = os.path.join(model_dir, "xgboost-model")
    print(f"Full model path: {model_path}")
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")
    
    try:
        with open(model_path, "rb") as f:
            model = pkl.load(f)
        print("Model loaded successfully.")

        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        raise e

def input_fn(request_body, content_type):
    """
    Parse input payload and return a DMatrix.
    """
    if content_type == "text/csv":
        data = io.StringIO(request_body.decode("utf-8") if isinstance(request_body, bytes) else request_body)
        df = pd.read_csv(data)
        return df.values
    else:
        raise ValueError(f"Unsupported content type: {content_type}")

def predict_fn(input_data, model):
    """
    Predict using the model.
    """
    prediction = model.predict(input_data)
    return prediction

def output_fn(prediction, accept):
    """
    Return prediction as a CSV-formatted string.
    """
    if accept == "text/csv":
        output = "\n".join(str(x) for x in prediction)
        return output, accept
    else:
        raise ValueError(f"Unsupported accept type: {accept}")

@app.route("/ping", methods=["GET"])
def ping():
    """
    Health check endpoint.
    """
    global model
    try:
        if model is None:
            model = model_fn("/opt/ml/model")
        return Response(response=json.dumps({"status": "ok"}), status=200, mimetype="application/json")
    except Exception as e:
        return Response(response=json.dumps({"error": str(e)}), status=500, mimetype="application/json")

@app.route("/invocations", methods=["POST"])
def invocations():
    """
    Inference endpoint.
    """
    global model
    try:
        if model is None:
            model = model_fn("/opt/ml/model")

        content_type = request.content_type or "text/csv"
        accept = request.headers.get("Accept", "text/csv")
        data = request.data

        input_data = input_fn(data, content_type)
        prediction = predict_fn(input_data, model)
        output, content_type = output_fn(prediction, accept)

        return Response(response=output, status=200, mimetype=content_type)

    except Exception as e:
        return Response(response=json.dumps({"error": str(e)}), status=500, mimetype="application/json")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
