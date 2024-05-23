import numpy as np
import torch
import os

def adjust_learning_rate(optimizer, epoch, args):
    # lr = args.learning_rate * (0.2 ** (epoch // 2))
    if args.lradj=='type1':
        lr_adjust = {epoch: args.learning_rate * (0.5 ** ((epoch-1) // 1))}
    elif args.lradj=='type2':
        lr_adjust = {
            2: 5e-5, 4: 1e-5, 6: 5e-6, 8: 1e-6, 
            10: 5e-7, 15: 1e-7, 20: 5e-8
        }
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        print('Updating learning rate to {}'.format(lr))

class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model, path, args):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path, args)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path, args)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path, args):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        model_save_path = os.path.join(path, 'checkpoint.pth')
        torch.save(model.state_dict(), model_save_path)

        # Save args
        args_save_path = os.path.join(path, 'args.pth')
        torch.save(args, args_save_path)

        self.val_loss_min = val_loss

class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

class StandardScaler():
    def __init__(self):
        self.mean = 0.
        self.std = 1.
    
    def fit(self, data):
        self.mean = data.mean(0)
        self.std = data.std(0)

    def transform(self, data):
        mean = torch.from_numpy(self.mean).type_as(data).to(data.device) if torch.is_tensor(data) else self.mean
        std = torch.from_numpy(self.std).type_as(data).to(data.device) if torch.is_tensor(data) else self.std
        return (data - mean) / std

    def inverse_transform(self, data):
        mean = torch.from_numpy(self.mean).type_as(data).to(data.device) if torch.is_tensor(data) else self.mean
        std = torch.from_numpy(self.std).type_as(data).to(data.device) if torch.is_tensor(data) else self.std
        if data.shape[-1] != mean.shape[-1]:
            mean = mean[-1:]
            std = std[-1:]
        return (data * std) + mean
    

def generate_and_convert_tensors(args):
    """
    This function generates example tensors based on provided arguments and converts them to nested Python lists.

    Parameters:
    args: Arguments including batch_size, seq_len, enc_in, pred_len, dec_in, and use_gpu.

    Returns:
    A dictionary where keys are tensor names and values are the nested Python lists.
    """

    # Create the device
    device = torch.device("cuda" if args.use_gpu else "cpu")

    # Generate inputs
    example_x_enc = torch.rand(args.batch_size, args.seq_len, args.enc_in).to(device).float()
    example_x_mark_enc = torch.rand(args.batch_size, args.seq_len, 1).to(device).float()
    example_x_dec = torch.rand(args.batch_size, args.pred_len, args.dec_in).to(device).float()
    example_x_mark_dec = torch.rand(args.batch_size, args.pred_len, 1).to(device).float()

    # Generate masks
    example_enc_self_mask = torch.rand(args.batch_size, args.seq_len, args.seq_len).to(device).float()
    example_dec_self_mask = torch.rand(args.batch_size, args.pred_len, args.pred_len).to(device).float()
    example_dec_enc_mask = torch.rand(args.batch_size, args.pred_len, args.seq_len).to(device).float()

    # Convert tensors to CPU before converting to nested Python lists
    example_x_enc = example_x_enc.cpu().numpy().tolist()
    example_x_mark_enc = example_x_mark_enc.cpu().numpy().tolist()
    example_x_dec = example_x_dec.cpu().numpy().tolist()
    example_x_mark_dec = example_x_mark_dec.cpu().numpy().tolist()
    example_enc_self_mask = example_enc_self_mask.cpu().numpy().tolist()
    example_dec_self_mask = example_dec_self_mask.cpu().numpy().tolist()
    example_dec_enc_mask = example_dec_enc_mask.cpu().numpy().tolist()

    return {
        "x_enc": example_x_enc,
        "x_mark_enc": example_x_mark_enc,
        "x_dec": example_x_dec,
        "x_mark_dec": example_x_mark_dec,
        "enc_self_mask": example_enc_self_mask,
        "dec_self_mask": example_dec_self_mask,
        "dec_enc_mask": example_dec_enc_mask,
    }