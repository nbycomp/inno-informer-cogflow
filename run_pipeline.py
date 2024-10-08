import argparse
import os
import sys
import torch
import pandas as pd
import numpy as np
import cogflow as cf
import logging
import kserve

def training(file_path: cf.input_path('parquet'), args: cf.input_path('json'))->str:
    import sys
    import os
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
        'Gtrace_5m': {'data': 'Gtrace_5m.csv', 'T': 'avg_cpu_usage', 'M': [10, 10, 10], 'S': [1, 1, 1], 'MS': [10, 10, 10]},
        'Gtrace_60m': {'data': 'Gtrace_60m.csv', 'T': 'avg_cpu_usage', 'M': [8, 8, 8], 'S': [1, 1, 1], 'MS': [7, 7, 1]},
    }

    if args.data in data_parser.keys():
        data_info = data_parser[args.data]
        args.data_path = data_info['data']
        args.target = data_info['T']
        args.enc_in, args.dec_in, args.c_out = data_info[args.features]

    args.s_layers = [int(s_l) for s_l in args.s_layers.replace(' ', '').split(',')]
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
    with cf.start_run('custom_model_run_informer') as run:
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
    return f"{run.info.artifact_uri}/{model_info.artifact_path}"


##################################################### PIPELINE ###########################################################

def preprocess(file_path: cf.input_path('CSV'), output_file: cf.output_path('parquet'), args: cf.output_path('json')):
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
    
    # Save args to a JSON file with specified and default values
    args_dict = {
        'experiment_name': 'dummy_exp_1',
        'model': 'informer',
        'data': 'Alibaba',
        'root_path': './data/',
        'data_path': 'processed_data.csv',
        'features': 'S',
        'target': 'avg_cpu_usage',
        'freq': 'm',
        'checkpoints': './checkpoints',
        'seq_len': 24,
        'label_len': 24,
        'pred_len': 12,
        'enc_in': 10,
        'dec_in': 10,
        'c_out': 10,
        'd_model': 128,
        'n_heads': 8,
        'e_layers': 2,
        'd_layers': 1,
        's_layers': '3,2,1',  # Use the default if None
        'd_ff': 512,
        'factor': 10,
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
        'num_workers': 0,
        'itr': 1,
        'train_epochs': 1,
        'batch_size': 32,
        'patience': 1,
        'learning_rate': 0.00001,
        'des': 'exp',
        'loss': 'mse',
        'lradj': 'type1',
        'use_amp': False,
        'inverse': False,
        'use_gpu': False,
        'gpu': 0,
        'use_multi_gpu': False,
        'devices': '0,1,2,3'
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
    try:
        import cogflow as cf
        cf.serve_model_v1(model_uri, name)
    except TypeError as e:
        print(f"Encountered TypeError: {e}")
        raise
    except Exception as e:
        print(f"Encountered an unexpected exception: {e}")
        raise



kserve_op=cf.create_component_from_func(
    func=serving,
    output_component_file='kserve-component.yaml',
    base_image='burntt/nby-cogflow-informer:latest',  # Example PyTorch image
    packages_to_install=[]
)

def getmodel(name):
    import cogflow as cf
    cf.get_model_url(name)
    

getmodel_op=cf.create_component_from_func(func=getmodel,
        output_component_file='kserve-component.yaml',
        base_image='burntt/bo-informer:v1',
        packages_to_install=[])

@cf.pipeline(name="informer-pipeline", description="Informer Time-Series Forecasting Pipeline")
def informer_pipeline(file, isvc):
    preprocess_task = preprocess_op(file='/data/Gtrace2019/Gtrace_5m.csv')
    
    train_task = training_op(
        file=preprocess_task.outputs['output'],
        args=preprocess_task.outputs['args']  # Pass the args output
    )
    
    kserve_task = kserve_op(model_uri=train_task.outputs['Output'], name=isvc)
    kserve_task.after(train_task)
    
    getmodel_task = getmodel_op(isvc)
    getmodel_task.after(kserve_task)

client = cf.client()
client.create_run_from_pipeline_func(
    informer_pipeline,
    arguments={
        "file": "/data/processed_data.csv",
        "isvc": "informer-serving-inference"
    }
)
