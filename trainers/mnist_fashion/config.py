
from imports import ClassEncoder, torch

class HyperParams(ClassEncoder):
    def __init__(self):
        super(HyperParams,self).__init__()
        self.nnParams()
        self.dataParams()
        
    def nnParams(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.epochs = 1
        self.batch = 64
        self.lr = 0.001
        
    def dataParams(self):
        self.img_size = (1,28,28)
        self.input_size = self.img_size[0]*self.img_size[1]
        self.output_size = 10
        self.train_len = 60000
        self.test_len = 10000

class NNTensors(ClassEncoder):
    def __init__(self, hparams):
        super(NNTensors,self).__init__()
        self.hparams = hparams

    def toDevice(self,nn_data:dict):
        for key,val in nn_data.items():
            if torch.is_tensor(val):
                nn_data[key] = val.to_device(self.device)
        self.addDict(**nn_data)
    
class StrAlias:
    data = "data"
    target = "target"
    pred = "pred"