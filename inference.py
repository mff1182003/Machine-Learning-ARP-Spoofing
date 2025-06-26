# inference.py
import joblib
import numpy as np
import pandas as pd
import os
import io

model = None
preprocessor = None

def model_fn(model_dir):
    global model
    global preprocessor
    model = joblib.load(os.path.join(model_dir, 'modelarp_custom_features_fixed_split.joblib'))
    preprocessor = joblib.load(os.path.join(model_dir, 'arppreprocessor_custom_features_fixed_split.joblib'))
    return model

def predict_fn(data, model):
    input_data = pd.read_csv(io.StringIO(data), header=None) # Giả sử dữ liệu đầu vào là CSV string
    processed_data = preprocessor.transform(input_data)
    prediction = model.predict(processed_data)
    return prediction.tolist()

def input_fn(request_body, request_content_type):
    if request_content_type == 'text/csv':
        return request_body
    elif request_content_type == 'application/json':
        import json
        return json.dumps(json.loads(request_body))
    else:
        raise ValueError("This predictor only supports CSV and JSON input")

def output_fn(prediction, content_type):
    if content_type == 'text/csv':
        return ','.join(map(str, prediction))
    elif content_type == 'application/json':
        import json
        return json.dumps(prediction)
    else:
        raise ValueError("This predictor only supports CSV and JSON output")