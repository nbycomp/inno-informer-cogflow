# 7.2.1 Time Series Forecasting Models (NBC)

## Introduction

This section details the integration of the Informer time series forecasting model into the Cognitive Framework (CF), emphasizing the model itself and the execution steps within the CF environment. The Informer model, as thoroughly described in Deliverable D3.1, is designed to efficiently handle long-sequence time series data using advanced attention mechanisms. By integrating Informer with the CF, we aim to enhance predictive capabilities in edge computing scenarios where computational resources are often limited.

## Purpose of the Integration and Overview of Steps

The primary purpose of this integration is to leverage the Informer's strengths within the CF to facilitate real-time forecasting in distributed systems. The integration streamlines the workflow from data ingestion to model deployment, allowing for automated preprocessing, training, and serving processes. This not only improves scalability and efficiency but also ensures that the predictive models are robust and adaptable to various edge computing environments.

The integration involves several key steps:

1. **Initial Setup in the CF Framework**: Establishing the development environment by creating a dedicated notebook within the CF.
2. **Pipeline Implementation**: Developing a pipeline that seamlessly connects data preprocessing, model training, and deployment stages.
3. **Execution and Deployment**: Running the pipeline within the CF to train the Informer model and deploy it as an inference service.

### Initial Setup in the CF Framework: Creation of the Notebook

The integration process begins with setting up a new project in the Cognitive Framework specifically for the Informer model. Within this project, a Jupyter notebook is created to serve as the central platform for development and execution. The notebook is configured with all necessary dependencies, including Python libraries such as PyTorch for deep learning and any Informer-specific modules.

Creating the notebook involves selecting the appropriate kernel that supports the required libraries and ensuring that the computational resources allocated are sufficient for both training and inference tasks. The interactive nature of the Jupyter notebook allows for real-time code execution, visualization of results, and iterative development, which is essential for fine-tuning the model and addressing any integration challenges.


## Pipeline

The core component of the integration is the pipeline that automates the entire workflow from data preprocessing to model deployment. The pipeline is designed to be modular, with each stage handling specific tasks that contribute to the overall functionality.

Below is a high-level representation of the pipeline:

# Pseudo-code of the Informer pipeline

```
def informer_pipeline(file, isvc):
    # Preprocess the input data
    preprocessed_data, args = preprocess(file)
    
    # Train the Informer model with the preprocessed data and arguments
    model_uri = training(preprocessed_data, args)
    
    # Deploy the trained model as an inference service
    serving_status = serving(model_uri, isvc)
    
    return serving_status
```

## Preprocessing

In the preprocessing stage, raw data is converted into a format suitable for model training. This involves reading the input data from CSV files, performing data cleaning to handle missing or inconsistent values, and conducting feature engineering to enhance the data's predictive power. The processed data is then saved in a parquet format, optimized for efficient data storage and retrieval during the training phase. Additionally, model hyperparameters and configurations are prepared and saved for use in the subsequent training stage.

## Training

The training stage focuses on configuring the Informer model with the appropriate hyperparameters and training it using the preprocessed data. Key aspects include defining hyperparameters such as sequence length, prediction length, learning rate, and model dimensions based on the specific requirements of the forecasting task. The model is then trained over a specified number of epochs. Metrics such as loss, accuracy, and validation scores are logged using the Cognitive Framework's logging capabilities to ensure transparency and reproducibility.

## Serving

Once the model has been trained, the serving stage involves deploying it to handle real-time inference requests. This includes packaging the trained model with all necessary dependencies and configurations, creating an inference service using technologies like Kubernetes and KServe within the CF, and exposing endpoints for sending data to the model and receiving predictions. The deployment is configured to handle scaling, load balancing, and monitoring, ensuring the model remains responsive and reliable in production environments.