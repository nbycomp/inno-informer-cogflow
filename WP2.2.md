# 7.2.1 Time Series Forecasting Models (NBC)

## Introduction

This section details the integration of the Informer time series forecasting model into the Cognitive Framework (CF), emphasizing the model itself and the execution steps within the CF environment. The Informer model, as thoroughly described in Deliverable D3.1, is designed to efficiently handle long-sequence time series data using advanced attention mechanisms. By integrating Informer with the CF, we aim to enhance predictive capabilities in edge computing scenarios where computational resources are often limited.

## Purpose of the Integration and Overview of Steps

The primary purpose of this integration is to leverage the Informer's strengths within the CF to facilitate real-time forecasting in distributed systems. The integration streamlines the workflow from data ingestion to model deployment, allowing for automated preprocessing, training, and serving processes. This not only improves scalability and efficiency but also ensures that the predictive models are robust and adaptable to various edge computing environments.

The integration involves several key steps:

1. **Initial Setup in the CF Framework**: Establishing the development environment by creating a dedicated notebook within the CF.
2. **Pipeline Implementation**: Developing a pipeline that seamlessly connects data preprocessing, model training, and deployment stages.
3. **Execution and Deployment**: Running the pipeline within the CF to train the Informer model and deploy it as an inference service.

### Informer Model: Why?

The Informer model is a specialized variant of the transformer architecture, designed to efficiently handle long-sequence time-series forecasting. It leverages sparse attention mechanisms to significantly reduce computational complexity, making it highly suitable for real-time applications in edge computing environments. This efficiency is achieved by focusing on the most relevant data points, ensuring the model captures long-range dependencies without the high computational cost of full attention.

Key innovations of the Informer include attention distilling, which further enhances memory efficiency by eliminating redundant entries in the attention matrix, and a generative-style decoder that outputs entire sequences in a single step. These features collectively accelerate the forecasting process and enable effective parallelization, crucial for time-critical use cases.

Despite its advantages in efficiency and speed, the Informer model's implementation complexity can be a challenge. Its specialized architecture requires careful tuning and ample training data to fully realize its potential. However, when properly configured, Informers offer a compelling balance of performance and computational efficiency, making them a powerful tool for long-sequence forecasting tasks.

### Initial Setup in the CF Framework: Creation of the Development Environment

The integration process begins with setting up a new project in the Cognitive Framework specifically for the Informer model. Within this project, a VSCode web environment is utilized, running a Docker image preloaded with all necessary dependencies, including Python libraries such as PyTorch for deep learning and any Informer-specific modules.

Setting up the environment involves selecting the appropriate Docker image that supports the required libraries and ensuring that the computational resources allocated are sufficient for both training and inference tasks. The interactive nature of the VSCode web environment allows for real-time code execution, visualization of results, and iterative development.


## Pipeline

The core component of the integration is the pipeline that automates the entire workflow from data preprocessing to model deployment. The pipeline is designed to be modular, with each stage handling specific tasks that contribute to the overall functionality.

Below is a high-level representation of the pipeline:

- **Preprocessing**: The pipeline begins with preprocessing the input data, where the data is prepared and necessary arguments are set.
- **Training**: The Informer model is trained using the preprocessed data and the specified arguments.
- **Deployment**: Finally, the trained model is deployed as an inference service, completing the pipeline process.

![image](https://github.com/user-attachments/assets/06a71f0e-ab79-4d4d-9403-17bed2f59afa)


## Preprocessing

In the preprocessing stage, raw data is converted into a format suitable for model training. This involves reading the input data from CSV files, performing data cleaning to handle missing or inconsistent values, and conducting feature engineering to enhance the data's predictive power. The processed data is then saved in a parquet format, optimized for efficient data storage and retrieval during the training phase. Additionally, model hyperparameters and configurations are prepared and saved for use in the subsequent training stage.

## Training

The training stage focuses on configuring the Informer model with the appropriate hyperparameters and training it using the preprocessed data. Key aspects include defining hyperparameters such as sequence length, prediction length, learning rate, and model dimensions based on the specific requirements of the forecasting task. The model is then trained over a specified number of epochs. Metrics such as loss, accuracy, and validation scores are logged using the Cognitive Framework's logging capabilities to ensure transparency and reproducibility.

## Serving

Once the model has been trained, the serving stage involves deploying it to handle real-time inference requests. This includes packaging the trained model with all necessary dependencies and configurations, creating an inference service using technologies like Kubernetes and KServe within the CF, and exposing endpoints for sending data to the model and receiving predictions. The deployment is configured to handle scaling, load balancing, and monitoring, ensuring the model remains responsive and reliable in production environments.
