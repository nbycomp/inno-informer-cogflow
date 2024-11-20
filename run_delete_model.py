import os
import urllib3
import warnings
from kubernetes import client, config

def delete_served_model(name: str) -> str:
    """
    Delete a served model via kubernetes API
    
    Args:
        name: Name of the inference service to delete
        
    Returns:
        str: Status message indicating success or failure
    """
    # Suppress InsecureRequestWarning
    warnings.filterwarnings('ignore', category=urllib3.exceptions.InsecureRequestWarning)
    
    # Disable SSL verification (use with caution)
    os.environ['CURL_CA_BUNDLE'] = ''
    
    try:
        print(f"Attempting to delete model: {name}")
        config.load_incluster_config()
        api_instance = client.CustomObjectsApi()
        
        # Get current namespace
        current_namespace = open("/var/run/secrets/kubernetes.io/serviceaccount/namespace").read()
        
        # Delete the InferenceService
        api_instance.delete_namespaced_custom_object(
            group="serving.kserve.io",
            version="v1beta1",
            namespace=current_namespace,
            plural="inferenceservices",
            name=name
        )
        
        return f"Successfully deleted model {name} using kubernetes API"
        
    except Exception as k8s_error:
        error_msg = f"Failed to delete model {name}. Errors: \n" \
                   f"Kubernetes error: {str(k8s_error)}"
        print(error_msg)
        return error_msg

# Example usage
if __name__ == "__main__":
    # Replace with your model name
    model_name = "informer-serving-inference"
    result = delete_served_model(model_name)
    print(result)
