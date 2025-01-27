import cogflow as cf

def training(file_path: cf.InputPath('parquet'), args: cf.InputPath('json'))->str:
    import sys
    import os
    import argparse
    import pandas as pd
    import torch
    import numpy as np
    import cogflow as cf 
    import json

    # Load the args from the JSON file
    with open(args, 'r') as f:
        args = argparse.Namespace(**json.load(f))

    if args.use_gpu and args.use_multi_gpu:
        args.devices = args.devices.replace(' ', '')
        device_ids = args.devices.split(',')
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]

    data_parser = {
        'alibaba_pod': {'data': 'processed_data.csv', 'T': 'cpu_utilization', 'M': [10, 10, 10], 'S': [1, 1, 1], 'MS': [10, 10, 10]},
    }

    if args.data in data_parser.keys():
        data_info = data_parser[args.data]
        args.data_path = data_info['data']
        args.target = data_info['T']
        args.enc_in, args.dec_in, args.c_out = data_info[args.features]

    args.detail_freq = args.freq
    args.freq = args.freq[-1:]

    print('Args in experiment:')
    print(args, '\n')

    # Log the system path before appending directories
    print("System path before appending directories:")
    print(sys.path)
    
    # Add root and necessary directories to sys.path
    sys.path.append('/')
    sys.path.append('/exp')
    sys.path.append('/models')
    sys.path.append('/utils')

    # Log the system path after appending directories
    print("System path after appending directories:")
    print(sys.path)

    # Log the contents of each directory for debugging
    directories_to_check = ['/exp', '/models', '/utils', '/data']
    for directory in directories_to_check:
        if os.path.isdir(directory):
            print(f"Contents of {directory}:")
            print(os.listdir(directory))
        else:
            print(f"{directory} is not a directory or does not exist.")

    # Attempt to import the required module
    try:
        from exp.exp_informer import Exp_Informer
        print("Import successful.")
    except ModuleNotFoundError as e:
        print(f"ModuleNotFoundError: {e}")
        return "Module import failed"

    Exp = Exp_Informer

    cf.autolog()
    cf.pytorch.autolog()
    experiment_id = cf.set_experiment(
        experiment_name="Custom Model Informer Time-Series",
    )

    # Replace the existing cf.start_run block with this:
    try:
        with cf.start_run(run_name='custom_model_run_informer', nested=True) as run:
            for ii in range(args.itr):
                setting = '{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_at{}_fc{}_eb{}_dt{}_mx{}_{}_{}'.format(
                    args.model, args.data, args.features, args.seq_len, args.label_len, args.pred_len, args.d_model, 
                    args.n_heads, args.e_layers, args.d_layers, args.d_ff, args.attn, args.factor, args.embed, 
                    args.distil, args.mix, args.des, ii
                )

                cf.log_param("seq_len", args.seq_len)
                cf.log_param("n_heads", args.n_heads)
                cf.log_param("enc_lay", args.e_layers)
                cf.log_param("pred_len", args.pred_len)
                cf.log_param("dec_lay", args.d_layers)

                exp = Exp(args)
                print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
                model = exp.train(setting)
                print('>>>>>>>end training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
                
                print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
                test_results = exp.test(setting)

                cf.log_metric("mae", test_results['mae'])
                cf.log_metric("mse", test_results['mse'])
                cf.log_metric("rmse", test_results['rmse'])
                cf.log_metric("r2", test_results['r2'])

                print('>>>>>>>predicting : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
                preds = exp.predict(setting, True)

                args_file_path = './args.txt'
                with open(args_file_path, 'w') as f:
                    for arg, value in vars(args).items():
                        f.write(f"{arg}={value}\n")
                artifacts = {"args.txt": args_file_path}

                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                model.to(device)


                print('ARGS before signature: ', args)

                example_x_enc = torch.rand(1, args.seq_len, args.enc_in).to(device).float()
                example_x_mark_enc = torch.rand(1, args.seq_len, 1).to(device).float()
                example_x_dec = torch.rand(1, args.pred_len, args.dec_in).to(device).float()
                example_x_mark_dec = torch.rand(1, args.pred_len, 1).to(device).float()
                inputs_example = (example_x_enc, example_x_mark_enc, example_x_dec, example_x_mark_dec)
                output_example = model(*inputs_example)

                inputs_example_cpu = tuple(tensor.cpu().detach().numpy() for tensor in inputs_example)
                output_example_cpu = output_example.cpu().detach().numpy()
                inputs_example_cpu_no_batch = tuple(input_array[0] for input_array in inputs_example_cpu)
                output_example_cpu_no_batch = output_example_cpu[0]

                inputs_combined = np.concatenate([input_array.flatten() for input_array in inputs_example_cpu_no_batch], axis=-1)
                input_df = pd.DataFrame(inputs_combined)

                try:
                    signature = cf.models.infer_signature(input_df, output_example_cpu_no_batch)
                    print('Inference Signature Correctly Saved!')
                except Exception as e:
                    print(f"Error inferring signature: {e}")
                    signature = None

                model_info = cf.pyfunc.log_model(
                    artifact_path='informer-alibaba-pod',
                    python_model=exp,
                    artifacts=artifacts,
                    pip_requirements=[],
                    input_example=input_df,
                    signature=signature
                )

                print(f"Run_id", run.info.run_id)
                print(f"Artifact_uri", run.info.artifact_uri)
                print(f"Artifact_path", run.info.artifact_uri)
                registered_models_list = cf.search_registered_models()
                print(registered_models_list)


                print('Returned String: ', f"{run.info.artifact_uri}/{model_info.artifact_path}")
    except Exception as e:
        print(f"Error starting run: {e}")
        # Create a new run if the previous one doesn't exist
        with cf.start_run(run_name='custom_model_run_informer') as run:
            for ii in range(args.itr):
                setting = '{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_at{}_fc{}_eb{}_dt{}_mx{}_{}_{}'.format(
                    args.model, args.data, args.features, args.seq_len, args.label_len, args.pred_len, args.d_model, 
                    args.n_heads, args.e_layers, args.d_layers, args.d_ff, args.attn, args.factor, args.embed, 
                    args.distil, args.mix, args.des, ii
                )

                cf.log_param("seq_len", args.seq_len)
                cf.log_param("n_heads", args.n_heads)
                cf.log_param("enc_lay", args.e_layers)
                cf.log_param("pred_len", args.pred_len)
                cf.log_param("dec_lay", args.d_layers)

                exp = Exp(args)
                print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
                model = exp.train(setting)
                print('>>>>>>>end training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
                
                print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
                test_results = exp.test(setting)

                cf.log_metric("mae", test_results['mae'])
                cf.log_metric("mse", test_results['mse'])
                cf.log_metric("rmse", test_results['rmse'])
                cf.log_metric("r2", test_results['r2'])

                print('>>>>>>>predicting : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
                preds = exp.predict(setting, True)

                args_file_path = './args.txt'
                with open(args_file_path, 'w') as f:
                    for arg, value in vars(args).items():
                        f.write(f"{arg}={value}\n")
                artifacts = {"args.txt": args_file_path}

                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                model.to(device)


                print('ARGS before signature: ', args)

                example_x_enc = torch.rand(1, args.seq_len, args.enc_in).to(device).float()
                example_x_mark_enc = torch.rand(1, args.seq_len, 1).to(device).float()
                example_x_dec = torch.rand(1, args.pred_len, args.dec_in).to(device).float()
                example_x_mark_dec = torch.rand(1, args.pred_len, 1).to(device).float()
                inputs_example = (example_x_enc, example_x_mark_enc, example_x_dec, example_x_mark_dec)
                output_example = model(*inputs_example)

                inputs_example_cpu = tuple(tensor.cpu().detach().numpy() for tensor in inputs_example)
                output_example_cpu = output_example.cpu().detach().numpy()
                inputs_example_cpu_no_batch = tuple(input_array[0] for input_array in inputs_example_cpu)
                output_example_cpu_no_batch = output_example_cpu[0]

                inputs_combined = np.concatenate([input_array.flatten() for input_array in inputs_example_cpu_no_batch], axis=-1)
                input_df = pd.DataFrame(inputs_combined)

                try:
                    signature = cf.models.infer_signature(input_df, output_example_cpu_no_batch)
                    print('Inference Signature Correctly Saved!')
                except Exception as e:
                    print(f"Error inferring signature: {e}")
                    signature = None

                model_info = cf.pyfunc.log_model(
                    artifact_path='informer-google-trace',
                    python_model=exp,
                    artifacts=artifacts,
                    pip_requirements=[],
                    input_example=input_df,
                    signature=signature
                )

                print(f"Run_id", run.info.run_id)
                print(f"Artifact_uri", run.info.artifact_uri)
                print(f"Artifact_path", run.info.artifact_uri)
                registered_models_list = cf.search_registered_models()
                print(registered_models_list)


                print('Returned String: ', f"{run.info.artifact_uri}/{model_info.artifact_path}")
    except Exception as e:
        print(f"Unexpected error: {e}")
        return "Error occurred during training"

    return f"{run.info.artifact_uri}/{model_info.artifact_path}"


##################################################### PIPELINE ###########################################################

def preprocess(file_path: cf.InputPath('CSV'), output_file: cf.OutputPath('parquet'), args: cf.OutputPath('json')):
    import pandas as pd
    import shutil
    import os
    import json

    # Read the CSV file and convert it to parquet format
    df = pd.read_csv(file_path, header=0, sep=";")
    
    # Serialize directory data into parquet file
    directory_data = {
        'exp': os.listdir('exp'),
        'models': os.listdir('models'),
        'utils': os.listdir('utils'),
        'data': os.listdir('data')
    }
    df['directory_data'] = directory_data
    df.to_parquet(output_file)
    
    # Save args to a JSON file with updated values
    args_dict = {
        'experiment_name': 'small_exp_1',
        'model': 'informer',
        'data': 'alibaba_pod',
        'root_path': './data/',
        'data_path': 'processed_data.csv',
        'features': 'S',
        'target': 'cpu_utilization',
        'freq': 'm',
        'checkpoints': './checkpoints',
        'seq_len': 12,
        'label_len': 12,
        'pred_len': 6,
        'enc_in': 1,
        'dec_in': 1,
        'c_out': 1,
        'd_model': 32,
        'n_heads': 4,
        'e_layers': 1,
        'd_layers': 1,
        'd_ff': 128,
        'factor': 5,
        'padding': 0,
        'distil': True,
        'dropout': 0.05,
        'attn': 'prob',
        'embed': 'timeF',
        'activation': 'gelu',
        'output_attention': False,
        'do_predict': False,
        'mix': True,
        'cols': None,
        'num_workers': 1,
        'itr': 1,
        'train_epochs': 1,
        'batch_size': 16,
        'patience': 1,
        'learning_rate': 0.00001,
        'des': 'small_exp',
        'loss': 'mse',
        'lradj': 'type1',
        'use_amp': False,
        'inverse': False,
        'use_gpu': False,
        'use_multi_gpu': False,
        'devices': '0'
    }

    with open(args, 'w') as f:
        json.dump(args_dict, f)

preprocess_op = cf.create_component_from_func(
    func=preprocess,
    output_component_file='preprocess-component.yaml',
    base_image='burntt/nby-cogflow-informer:latest',
    packages_to_install=[]
)


training_op = cf.create_component_from_func(
    func=training,
    output_component_file='train-component.yaml',
    base_image='burntt/nby-cogflow-informer:latest',  # Example PyTorch image
    packages_to_install=[]
)

def serving(model_uri, name):
    import cogflow as cf
    import os
    import urllib3
    import warnings
    
    # Suppress InsecureRequestWarning
    warnings.filterwarnings('ignore', category=urllib3.exceptions.InsecureRequestWarning)
    
    # Disable SSL verification (use with caution)
    os.environ['CURL_CA_BUNDLE'] = ''
    
    try:
        print(f"Serving model from URI: {model_uri}")
        
        # Attempt to serve the model without the verify_ssl parameter
        cf.serve_model_v1(model_uri, name)
        
        print(f"Model served successfully with name: {name}")
    except Exception as e:
        print(f"Error during model serving: {str(e)}")
        
        # If the first attempt fails, try an alternative method
        try:
            from kubernetes import client, config
            
            print("Attempting alternative serving method...")
            config.load_incluster_config()
            api_instance = client.CustomObjectsApi()
            
            inference_service = {
                "apiVersion": "serving.kserve.io/v1beta1",
                "kind": "InferenceService",
                "metadata": {"name": name},
                "spec": {
                    "predictor": {
                        "model": {
                            "modelFormat": {"name": "mlflow"},
                            "storageUri": model_uri
                        }
                    }
                }
            }
            
            # Try to create the InferenceService in the current namespace
            current_namespace = open("/var/run/secrets/kubernetes.io/serviceaccount/namespace").read()
            
            api_instance.create_namespaced_custom_object(
                group="serving.kserve.io",
                version="v1beta1",
                namespace=current_namespace,
                plural="inferenceservices",
                body=inference_service
            )
            
            print(f"Model served successfully using alternative method with name: {name}")
        except Exception as alt_e:
            print(f"Error during alternative serving method: {str(alt_e)}")
            raise
    
    return "Model serving completed"

kserve_op = cf.create_component_from_func(
    func=serving,
    output_component_file='kserve-component.yaml',
    base_image='burntt/nby-cogflow-informer:latest',
    packages_to_install=['kubernetes']
)

def getmodel(name: str) -> str:
    import cogflow as cf
    import os
    import warnings
    
    # Return the model URL
    return cf.get_model_url(name)

getmodel_op = cf.create_component_from_func(
    func=getmodel,
    output_component_file='getmodel-component.yaml',
    base_image='burntt/nby-cogflow-informer:latest',  # Added base image that contains cogflow
    packages_to_install=[]  # Specify any additional packages if needed
)

@cf.pipeline(name="informer-pipeline", description="Informer Time-Series Forecasting Pipeline")
def informer_pipeline(file, isvc):
    preprocess_task = preprocess_op(file=file)
    
    train_task = training_op(
        file=preprocess_task.outputs['output'],
        args=preprocess_task.outputs['args']
    )
    
    serve_task = kserve_op(model_uri=train_task.output, name=isvc)
    serve_task.after(train_task)
    
    getmodel_task = getmodel_op(name=isvc)
    getmodel_task.after(serve_task)

client = cf.client()
client.create_run_from_pipeline_func(
    informer_pipeline,
    arguments={
        "file": "/data/processed_data.csv",
        "isvc": "informer-serving-inference"
    }
)
