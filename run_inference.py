import requests
import json
import numpy as np
import pandas as pd

def create_inference_request(seq_len=12, pred_len=6, enc_in=1, dec_in=1):
    """
    Creates a properly formatted inference request for the Informer model.
    
    Args:
        seq_len (int): Sequence length for input
        pred_len (int): Prediction length
        enc_in (int): Number of input features for encoder
        dec_in (int): Number of input features for decoder
    """
    # Create random example data matching the model's expected input format
    x_enc = np.random.rand(seq_len, enc_in)
    x_mark_enc = np.random.rand(seq_len, 1)
    x_dec = np.random.rand(pred_len, dec_in)
    x_mark_dec = np.random.rand(pred_len, 1)
    
    # Flatten and concatenate all inputs as per the model's signature
    inputs_combined = np.concatenate([
        x_enc.flatten(),
        x_mark_enc.flatten(),
        x_dec.flatten(),
        x_mark_dec.flatten()
    ])
    
    # Convert to DataFrame and then to list for JSON serialization
    input_df = pd.DataFrame(inputs_combined).T
    input_data = input_df.values.tolist()[0]
    
    # Create the inference request JSON
    inference_request = {
        "inputs": [
            {
                "name": "input_0",
                "shape": [1, len(input_data)],
                "datatype": "FP32",
                "data": input_data
            }
        ]
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
            verify=False  # Note: In production, you should handle SSL verification properly
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
    # Create the inference request
    request_data = create_inference_request()
    
    # Model URL (replace with your actual model URL)
    model_url = "http://informer-serving-inference.adminh.svc.cluster.local"
    
    # Run inference
    result = run_inference(model_url, request_data)
    
    if result:
        print("Inference successful!")
        print("Predictions:", result)
    else:
        print("Inference failed!")

# Example of how to use the prediction results
"""
# The response will be in this format:
{
    "outputs": [
        {
            "name": "output_0",
            "shape": [1, pred_len],  # pred_len is your prediction length (6 in your case)
            "datatype": "FP32",
            "data": [...]  # Predicted values
        }
    ]
}
"""
