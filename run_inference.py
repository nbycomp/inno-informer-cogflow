import requests
import json
import numpy as np
import pandas as pd

def create_inference_request(seq_len=12, pred_len=6, enc_in=1, dec_in=1):
    """
    Creates a properly formatted inference request for the Informer model.
    
    Args:
        seq_len (int): Sequence length for encoder input (12)
        pred_len (int): Prediction length (6)
        enc_in (int): Number of input features for encoder (1)
        dec_in (int): Number of input features for decoder (1)
    """
    # Create the four input tensors with correct shapes
    x_enc = np.random.rand(1, seq_len, enc_in).astype(np.float32)
    x_mark_enc = np.random.rand(1, seq_len, 1).astype(np.float32)  # 1 feature for marking
    x_dec = np.random.rand(1, pred_len, dec_in).astype(np.float32)
    x_mark_dec = np.random.rand(1, pred_len, 1).astype(np.float32)  # 1 feature for marking
    
    # Create the inference request JSON following the example format
    inference_request = {
        "inputs": [
            {
                "name": "input0",
                "datatype": "FP32",
                "shape": [1, seq_len, enc_in],
                "parameters": {"content_type": "np"},
                "data": x_enc.tolist()
            },
            {
                "name": "input1",
                "datatype": "FP32",
                "shape": [1, seq_len, 1],  # Changed to 1 marking feature
                "parameters": {"content_type": "np"},
                "data": x_mark_enc.tolist()
            },
            {
                "name": "input2",
                "datatype": "FP32",
                "shape": [1, pred_len, dec_in],
                "parameters": {"content_type": "np"},
                "data": x_dec.tolist()
            },
            {
                "name": "input3",
                "datatype": "FP32",
                "shape": [1, pred_len, 1],  # Changed to 1 marking feature
                "parameters": {"content_type": "np"},
                "data": x_mark_dec.tolist()
            }
        ],
        "parameters": {"content_type": "np"}
    }
    
    return inference_request

def run_inference(model_url, request_data):
    """
    Sends an inference request to the deployed model.
    
    Args:
        model_url (str): URL of the deployed model
        request_data (dict): Formatted inference request data
    """
    headers = {
        "Content-Type": "application/json"
    }
    
    try:
        # Send POST request to the model's inference endpoint
        response = requests.post(
            f"{model_url}/v2/models/informer-serving-inference/infer",
            headers=headers,
            data=json.dumps(request_data),
            verify=False
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            print(f"Error: Received status code {response.status_code}")
            print(f"Response: {response.text}")
            return None
            
    except Exception as e:
        print(f"Error during inference request: {str(e)}")
        return None

if __name__ == "__main__":
    # Create the inference request with the correct dimensions from args_dict
    request_data = create_inference_request(
        seq_len=12,    # From args_dict['seq_len']
        pred_len=6,    # From args_dict['pred_len']
        enc_in=1,      # From args_dict['enc_in']
        dec_in=1       # From args_dict['dec_in']
    )
    
    # Model URL (replace with your actual model URL)
    model_url = "http://informer-serving-inference.adminh.svc.cluster.local"
    
    # Run inference
    result = run_inference(model_url, request_data)
    
    if result:
        print("Inference successful!")
        print("Predictions:", result)
        # Expected output shape: [1, 6, 1] (batch_size, pred_len, output_features)
    else:
        print("Inference failed!")

"""
Expected response format:
{
    "outputs": [
        {
            "name": "output0",
            "shape": [1, 6, 1],  # batch_size, pred_len, output_features
            "datatype": "FP32",
            "data": [...]  # Predicted values
        }
    ]
}
"""
