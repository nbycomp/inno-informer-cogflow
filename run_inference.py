import requests
import json
import numpy as np
import cogflow
from typing import Optional, Dict, Any

def get_model_url(model_name: str = "informer-serving-inference", namespace: str = "ver") -> str:
    """
    Get the URL for the deployed model using cogflow.
    
    Args:
        model_name (str): Name of the deployed model
        namespace (str): Kubernetes namespace where model is deployed
        
    Returns:
        str: The complete URL for the model endpoint
    """
    print(f"Attempting to get URL for model: {model_name} in namespace: {namespace}")
    try:
        # Try using cogflow first
        url = cogflow.get_model_url(model_name=model_name)
        print(f"Successfully retrieved model URL via cogflow: {url}")
        return url
    except Exception as e:
        print(f"Cogflow URL retrieval failed: {str(e)}")
        # Fallback to constructing URL manually
        url = f"http://{model_name}.{namespace}.svc.cluster.local"
        print(f"Using constructed URL: {url}")
        return url

def verify_model_ready(model_url: str) -> bool:
    """
    Verify if the model is ready to accept requests.
    """
    print(f"\nVerifying model readiness at {model_url}")
    try:
        # Try to get model metadata
        response = requests.get(f"{model_url}/v2/models/informer-serving-inference")
        if response.status_code == 200:
            print("Model is ready!")
            return True
        else:
            print(f"Model not ready. Status code: {response.status_code}")
            print(f"Response: {response.text}")
            return False
    except Exception as e:
        print(f"Error checking model readiness: {str(e)}")
        return False

def create_inference_request(data: Optional[np.ndarray] = None) -> Dict[str, Any]:
    """
    Creates a properly formatted inference request for the Informer model.
    """
    print("\nCreating inference request...")
    
    # If no data provided, create sample data
    if data is None:
        seq_len, pred_len = 12, 6
        enc_in, dec_in = 1, 1
        
        # Create sample input tensors
        x_enc = np.random.rand(1, seq_len, enc_in).astype(np.float32)
        x_mark_enc = np.random.rand(1, seq_len, 1).astype(np.float32)
        x_dec = np.random.rand(1, pred_len, dec_in).astype(np.float32)
        x_mark_dec = np.random.rand(1, pred_len, 1).astype(np.float32)
        
        print(f"Created sample tensors with shapes:")
        print(f"x_enc: {x_enc.shape}")
        print(f"x_mark_enc: {x_mark_enc.shape}")
        print(f"x_dec: {x_dec.shape}")
        print(f"x_mark_dec: {x_mark_dec.shape}")
    
    inference_request = {
        "inputs": [
            {
                "name": "input0",
                "datatype": "FP32",
                "shape": x_enc.shape,
                "data": x_enc.tolist()
            },
            {
                "name": "input1",
                "datatype": "FP32",
                "shape": x_mark_enc.shape,
                "data": x_mark_enc.tolist()
            },
            {
                "name": "input2",
                "datatype": "FP32",
                "shape": x_dec.shape,
                "data": x_dec.tolist()
            },
            {
                "name": "input3",
                "datatype": "FP32",
                "shape": x_mark_dec.shape,
                "data": x_mark_dec.tolist()
            }
        ]
    }
    
    print("Created inference request structure")
    return inference_request

if __name__ == "__main__":
    print("Starting inference process...")
    
    try:
        # Step 1: Get model URL
        model_url = get_model_url(namespace="ver")
        print(f"Model URL: {model_url}")
        
        # Step 2: Verify model is ready
        if not verify_model_ready(model_url):
            print("Model is not ready. Please check deployment status.")
            exit(1)
        
        # Step 3: Create and send inference request
        request_data = create_inference_request()
        
        print("\nSending inference request...")
        response = requests.post(
            f"{model_url}/v2/models/informer-serving-inference/infer",
            headers={"Content-Type": "application/json"},
            data=json.dumps(request_data)
        )
        
        if response.status_code == 200:
            result = response.json()
            print("\nInference successful!")
            print("Predictions:", json.dumps(result, indent=2))
        else:
            print(f"\nInference failed with status code: {response.status_code}")
            print(f"Error response: {response.text}")
            
    except Exception as e:
        print(f"\nError during inference process: {str(e)}")
