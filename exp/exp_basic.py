import os
import torch
import cogflow as cf

class Exp_Basic(cf.pyfunc.PythonModel):
    def __init__(self, args):
        self.args = args
        self.device = self._acquire_device()
        self.model = self._build_model().to(self.device)

    def _build_model(self):
        raise NotImplementedError
        return None
    
    def _acquire_device(self):
        if self.args.use_gpu:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(self.args.gpu) if not self.args.use_multi_gpu else self.args.devices
            device = torch.device('cuda:{}'.format(self.args.gpu))
            print('Use GPU: cuda:{}'.format(self.args.gpu))
        else:
            device = torch.device('cpu')
            print('Use CPU')
        return device

    def _get_data(self):
        pass

    def vali(self):
        pass

    def train(self):
        pass

    def test(self):
        pass

    def predict(self):
        pass

    def save(self, filename):
        torch.save(self.model.state_dict(), filename)
        return filename

    def load(self, filename):
        self.model.load_state_dict(torch.load(filename))
    
    def load_context(self, context):
        model_file_path = context.artifacts["model"]
        self.load(model_file_path)