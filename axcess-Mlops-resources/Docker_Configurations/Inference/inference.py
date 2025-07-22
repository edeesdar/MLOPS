import os
import io
import json
import pandas as pd
import xgboost as xgb
from flask import Flask, request, Response
 
app = Flask(__name__)
model = None
 
# Expected model features (case-sensitive)
EXPECTED_COLUMNS = [
    "pickup_hour", "pickup_day", "pickup_month", "dropoff_hour",
    "vendorid", "store_and_fwd_flag", "ratecodeid", "pulocationid",
    "dolocationid", "payment_type"
]
 
# Mapping from input CSV to training column names
# This RENAME_MAP is useful if your input CSV has headers that differ from EXPECTED_COLUMNS
# For a raw CSV with no header (as per your input_fn), the order is critical.
RENAME_MAP = {
    'vendorid': 'VendorID',
    'store_and_fwd_flag': 'store_and_fwd_flag',
    'ratecodeid': 'RatecodeID',
    'pulocationid': 'PULocationID',
    'dolocationid': 'DOLocationID',
    'pickup_hour': 'pickup_hour',
    'pickup_day': 'pickup_day',
    'pickup_month': 'pickup_month',
    'dropoff_hour': 'dropoff_hour',
    'payment_type': 'payment_type'
}
 
def model_fn(model_dir):
    """
    Load the XGBoost Booster model from disk.
    """
    model_path = os.path.join(model_dir, "xgboost-model")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")
 
    booster = xgb.Booster()
    booster.load_model(model_path)
    print(" Model loaded successfully.")

    print("\n===== Booster Attributes =====")
    print(booster.attributes())  # hyperparameters and training attributes

    print("\n===== Feature Importance (gain) =====")
    print(booster.get_score(importance_type='gain'))  # importance by gain

    print("\n===== First Tree in Model =====")
    dump = booster.get_dump()
    print(dump[0] if dump else "No trees found in model.")

    print(f"\n Model contains {len(dump)} trees.")
 
    return booster
 
def input_fn(request_body, content_type):
    if content_type == "text/csv":
        data = io.StringIO(request_body.decode("utf-8") if isinstance(request_body, bytes) else request_body)
        df = pd.read_csv(data, header=0)

        # Check that all expected columns are present
        missing_cols = [col for col in EXPECTED_COLUMNS if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns for model inference: {missing_cols}")

        # Select only expected columns in order
        df = df[EXPECTED_COLUMNS]

        # Convert to numeric and handle missing values
        df = df.apply(pd.to_numeric, errors='coerce').fillna(0).astype(float)

        print(f"Processed input shape: {df.shape}")
        print(df.head(10))
        print('df columns')
        print(df.columns)
        return df
    else:
        raise ValueError(f"Unsupported content type: {content_type}")
 
def predict_fn(input_data, model):
    """
    Makes predictions using the loaded XGBoost model.
    The input_data is expected to have the correct column names from input_fn.
    """
    # Ensure input_data contains only the expected features in the correct order
    # This is crucial because DMatrix uses column order if feature_names are not explicitly set.
    # We ensure `input_data` passed to DMatrix only has the `EXPECTED_COLUMNS`
    # and they are in the correct order.
    input_data_ordered = input_data[EXPECTED_COLUMNS]

    print("===== Data Passed to Model =====")

    print(input_data_ordered.head(10))
 
    # Create DMatrix without explicitly setting feature_names,
    # as the model was loaded without them and validate_features=False is used.
    dmatrix = xgb.DMatrix(input_data_ordered)
 
    # Disable feature name validation (as per your original code)
    prediction = model.predict(dmatrix, validate_features=False)

    print("===== Predictions =====")
    print(prediction[:10]) 
 
    return prediction
 
def output_fn(prediction, accept):
    """
    Format prediction output.
    """
    if accept == "text/csv":
        return "\n".join(str(x) for x in prediction), "text/csv"
    else:
        raise ValueError(f" Unsupported accept type: {accept}")
 
@app.route("/ping", methods=["GET"])
def ping():
    global model
    try:
        if model is None:
            model = model_fn("/opt/ml/model")
        return Response(response=json.dumps({"status": "ok"}), status=200, mimetype="application/json")
    except Exception as e:
        return Response(response=json.dumps({"error": str(e)}), status=500, mimetype="application/json")
 
@app.route("/invocations", methods=["POST"])
def invocations():
    global model
    try:
        if model is None:
            model = model_fn("/opt/ml/model")
 
        content_type = request.content_type or "text/csv"
        accept = request.headers.get("Accept", "text/csv")
        data = request.data
 
        input_df = input_fn(data, content_type)
        prediction = predict_fn(input_df, model)
        output, content_type = output_fn(prediction, accept)
 
        return Response(response=output, status=200, mimetype=content_type)
 
    except Exception as e:
        return Response(response=json.dumps({"error": str(e)}), status=500, mimetype="application/json")
 
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)