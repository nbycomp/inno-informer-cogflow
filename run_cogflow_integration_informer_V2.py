import argparse
import os
import torch
import pandas as pd
import numpy as np
import cogflow as cf
import logging
import sys
import kserve


print(sys.executable)

############### Arguments from Shell ##################
parser = argparse.ArgumentParser(description='[Informer] Long Sequences Forecasting')

parser.add_argument('--experiment_name', type=str, default='default_exp', help='Name of the MLFlow experiment')
parser.add_argument('--model', type=str, required=True, default='informer', help='model of experiment, options: [informer, informerstack, informerlight(TBD)]')
parser.add_argument('--data', type=str, required=True, default='ETTh1', help='data')
parser.add_argument('--root_path', type=str, default='./ETDataset/ETT-small/', help='root path of the data file')
parser.add_argument('--data_path', type=str, default='ETTh1.csv', help='data file')    
parser.add_argument('--features', type=str, default='M', help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
parser.add_argument('--freq', type=str, default='h', help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')
parser.add_argument('--seq_len', type=int, default=96, help='input sequence length of Informer encoder')
parser.add_argument('--label_len', type=int, default=48, help='start token length of Informer decoder')
parser.add_argument('--pred_len', type=int, default=24, help='prediction sequence length')
parser.add_argument('--enc_in', type=int, default=7, help='encoder input size')
parser.add_argument('--dec_in', type=int, default=7, help='decoder input size')
parser.add_argument('--c_out', type=int, default=7, help='output size')
parser.add_argument('--d_model', type=int, default=512, help='dimension of model')
parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
parser.add_argument('--s_layers', type=str, default='3,2,1', help='num of stack encoder layers')
parser.add_argument('--d_ff', type=int, default=2048, help='dimension of fcn')
parser.add_argument('--factor', type=int, default=5, help='probsparse attn factor')
parser.add_argument('--padding', type=int, default=0, help='padding type')
parser.add_argument('--distil', action='store_false', help='whether to use distilling in encoder, using this argument means not using distilling', default=True)
parser.add_argument('--dropout', type=float, default=0.05, help='dropout')
parser.add_argument('--attn', type=str, default='prob', help='attention used in encoder, options:[prob, full]')
parser.add_argument('--embed', type=str, default='timeF', help='time features encoding, options:[timeF, fixed, learned]')
parser.add_argument('--activation', type=str, default='gelu', help='activation')
parser.add_argument('--output_attention', action='store_true', help='whether to output attention in ecoder')
parser.add_argument('--do_predict', action='store_true', help='whether to predict unseen future data')
parser.add_argument('--mix', action='store_false', help='use mix attention in generative decoder', default=True)
parser.add_argument('--cols', type=str, nargs='+', help='certain cols from the data files as the input features')
parser.add_argument('--num_workers', type=int, default=0, help='data loader num workers')
parser.add_argument('--itr', type=int, default=2, help='experiments times')
parser.add_argument('--train_epochs', type=int, default=6, help='train epochs')
parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
parser.add_argument('--patience', type=int, default=3, help='early stopping patience')
parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
parser.add_argument('--des', type=str, default='test', help='exp description')
parser.add_argument('--loss', type=str, default='mse', help='loss function')
parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')
parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)
parser.add_argument('--inverse', action='store_true', help='inverse output data', default=False)
parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
parser.add_argument('--gpu', type=int, default=0, help='gpu')
parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multile gpus')

args = parser.parse_args()
args.use_gpu = False

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

def preprocess(file_path: cf.input_path('CSV'), output_file: cf.output_path('parquet')):
    df = pd.read_csv(file_path, header=0, sep=";")
    df.columns = [c.lower().replace(' ', '_') for c in df.columns]
    df.to_parquet(output_file)

def training(file_path: cf.input_path('parquet')) -> str:
    import sys
    import os
    import pandas as pd
    import torch
    import numpy as np
    import shutil
    
    # Add directories to the system path
    sys.path.append('./exp')
    sys.path.append('./models')
    sys.path.append('./utils')
    sys.path.append('./data')

    from exp.exp_informer import Exp_Informer

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
    return f"{run.info.artifact_uri}/{model_info.artifact_path}"


##################################################### PIPELINE ###########################################################

def preprocess(file_path: cf.input_path('CSV'),
              output_file: cf.output_path('parquet')):
    import pandas as pd
    import shutil
    import os

    # Read the CSV file and convert it to parquet format
    df = pd.read_csv(file_path, header=0, sep=";")
    df.to_parquet(output_file)

    # Define the directories to copy
    directories_to_copy = ['exp', 'models', 'utils', 'data']
    
    # Copy each directory to the output location
    for directory in directories_to_copy:
        if os.path.exists(directory):
            shutil.copytree(directory, os.path.join(os.path.dirname(output_file), directory))
        else:
            print(f"Directory {directory} does not exist and will not be copied.")


preprocess_op = cf.create_component_from_func(
    func=preprocess,
    output_component_file='preprocess-component.yaml',
    base_image='burntt/bo-informer:v1',  # Example PyTorch image
    packages_to_install=[]
)


training_op = cf.create_component_from_func(
    func=training,
    output_component_file='train-component.yaml',
    base_image='burntt/bo-informer:v1',  # Example PyTorch image
    packages_to_install=[]
)

def serving(model_uri, name):
    cf.serve_model_v1(model_uri, name)

kserve_op=cf.create_component_from_func(
    func=serving,
    output_component_file='kserve-component.yaml',
    base_image='burntt/bo-informer:v1',  # Example PyTorch image
    packages_to_install=[]
)

def getmodel(name):
    cf.get_model_url(name)

getmodel_op=cf.create_component_from_func(func=getmodel,
        output_component_file='kserve-component.yaml',
        base_image='burntt/bo-informer:v1',
        packages_to_install=[])

@cf.pipeline(name="informer-pipeline", description="Informer Time-Series Forecasting Pipeline")
def informer_pipeline(url, isvc):
    preprocess_task = preprocess_op(file='./data/Gtrace2019/Gtrace_5m.csv')
    
    train_task = training_op(file=preprocess_task.outputs['output'])
    train_task = train_task.AddModelAccess()
    
    kserve_task = kserve_op(model_uri=train_task.output, name=isvc)
    kserve_task = kserve_task.AddModelAccess()
    kserve_task.after(train_task)
    
    getmodel_task = getmodel_op(isvc)
    getmodel_task.after(kserve_task)

client = cf.client()
client.create_run_from_pipeline_func(
    informer_pipeline,
    arguments={
        "url": "./data/Gtrace2019/Gtrace_5m.csv",
        "isvc": "sample-final-bola33"
    }
)