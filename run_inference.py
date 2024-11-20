import requests
import json
import numpy as np
import cogflow
import time
from typing import Optional, Dict, Any

def get_model_url(model_name: str = "informer-serving-inference") -> str:
    """
    Get the URL for the deployed model using cogflow.
    
    Args:
        model_name (str): Name of the deployed model
        
    Returns:
        str: The complete URL for the model endpoint
    """
    print(f"Attempting to get URL for model: {model_name}")
    try:
        url = cogflow.get_model_url(model_name=model_name)
        print(f"Successfully retrieved model URL: {url}")
        return url
    except Exception as e:
        print(f"Error getting model URL: {str(e)}")
        raise

def get_model_signature(model_url: str) -> Optional[Dict[str, Any]]:
    """
    Get the model signature to verify input/output format.
    
    Args:
        model_url (str): The base URL for the model
        
    Returns:
        dict: Model signature if successful, None otherwise
    """
    print(f"\nStep 1: Getting model signature from {model_url}")
    try:
        # Construct signature endpoint
        signature_url = f"{model_url}/v2/models/informer-serving-inference"
        print(f"Requesting signature from: {signature_url}")
        
        # Make request
        response = requests.get(signature_url)
        print(f"Response status code: {response.status_code}")
        
        if response.status_code == 200:
            signature = response.json()
            print("Successfully retrieved model signature:")
            print(json.dumps(signature, indent=2))
            return signature
        else:
            print(f"Error response: {response.text}")
            return None
            
    except Exception as e:
        print(f"Exception while getting signature: {str(e)}")
        return None

def create_inference_request(seq_len: int = 12, 
                           pred_len: int = 6, 
                           enc_in: int = 1, 
                           dec_in: int = 1) -> Dict[str, Any]:
    """
    Creates a properly formatted inference request for the Informer model.
    
    Args:
        seq_len (int): Sequence length for encoder input
        pred_len (int): Prediction length
        enc_in (int): Number of input features for encoder
        dec_in (int): Number of input features for decoder
        
    Returns:
        dict: Formatted inference request
    """
    print(f"\nStep 2: Creating inference request with parameters:")
    print(f"  seq_len: {seq_len}")
    print(f"  pred_len: {pred_len}")
    print(f"  enc_in: {enc_in}")
    print(f"  dec_in: {dec_in}")
    
    # Create input tensors
    x_enc = np.random.rand(1, seq_len, enc_in).astype(np.float32)
    x_mark_enc = np.random.rand(1, seq_len, 1).astype(np.float32)
    x_dec = np.random.rand(1, pred_len, dec_in).astype(np.float32)
    x_mark_dec = np.random.rand(1, pred_len, 1).astype(np.float32)
    
    print("\nCreated input tensors with shapes:")
    print(f"  x_enc: {x_enc.shape}")
    print(f"  x_mark_enc: {x_mark_enc.shape}")
    print(f"  x_dec: {x_dec.shape}")
    print(f"  x_mark_dec: {x_mark_dec.shape}")
    
    # Create request structure
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
                "shape": [1, seq_len, 1],
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
                "shape": [1, pred_len, 1],
                "parameters": {"content_type": "np"},
                "data": x_mark_dec.tolist()
            }
        ],
        "parameters": {"content_type": "np"}
    }
    
    print("\nCreated inference request structure")
    return inference_request

def run_inference(model_url: str, request_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Sends an inference request to the deployed model.
    
    Args:
        model_url (str): The base URL for the model
        request_data (dict): The formatted inference request
        
    Returns:
        dict: Model predictions if successful, None otherwise
    """
    print(f"\nStep 3: Sending inference request to {model_url}")
    
    # Setup headers
    headers = {
        "Content-Type": "application/json"
    }
    print("Request headers:", headers)
    
    try:
        # Construct inference endpoint
        inference_url = f"{model_url}/v2/models/informer-serving-inference/infer"
        print(f"Sending POST request to: {inference_url}")
        
        # Make request
        response = requests.post(
            inference_url,
            headers=headers,
            data=json.dumps(request_data)
        )
        print(f"Response status code: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print("Successfully received prediction:")
            print(json.dumps(result, indent=2))
            return result
        else:
            print(f"Error response: {response.text}")
            return None
            
    except Exception as e:
        print(f"Exception during inference: {str(e)}")
        return None

if __name__ == "__main__":
    print("Starting inference process...")
    
    try:
        # Step 1: Get model URL
        print("\nGetting model URL...")
        model_url = get_model_url()
        
        # Step 2: Get model signature
        signature = get_model_signature(model_url)
        if not signature:
            print("Failed to get model signature, but continuing...")
        
        # Step 3: Create inference request
        request_data = create_inference_request(
            seq_len=12,    # From your model configuration
            pred_len=6,    # From your model configuration
            enc_in=1,      # From your model configuration
            dec_in=1       # From your model configuration
        )
        
        # Step 4: Run inference
        result = run_inference(model_url, request_data)
        
        if result:
            print("\nInference process completed successfully!")
            print("Final predictions:", json.dumps(result, indent=2))
        else:
            print("\nInference process failed!")
            
    except Exception as e:
        print(f"\nFatal error during inference process: {str(e)}")

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
