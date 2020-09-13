from data_loader.mnist_fashion import train_loader, test_loader
from dnn_models import fcc_v1
from . import hparams, sa, NNTensors
from imports import nnloss, nnoptims, nn

class FashionMnist:
    def __init__(self):
        self.model = fcc_v1.FCV1(
            input_size=hparams.input_size,
            out_class_num=hparams.output_size
        ).to(hparams.device)
        self.criterion = nnloss.crossEntropy()
        self.optim = nnoptims.Adam(
            params = self.model.parameters(),
            lr = hparams.lr
        )
        self.fmt = NNTensors()  

    def update(self):
        self.optim.zero_grad()
        self.fmt.loss.backward()
        self.optim.step()                

    def nnTrainer(self):
        for epoch in range(hparams.epochs):
            self.processEpoch()
        
    def processEpoch(self):
        for batch_idx, (data, target) in enumerate(train_loader):
            self.fmt.toDevice(
                nn_data={
                    sa.data:data,
                    sa.target:target
                }
            )
            self.fmt.data = self.fmt.data.reshape(data.shape[0],-1)
            self.fmt.pred = self.model(data)
            self.fmt.loss = self.criterion(self.fmt.pred, self.fmt.target)
            self.update()
                
            

