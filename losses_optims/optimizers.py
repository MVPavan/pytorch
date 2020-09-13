from imports import optim
class NNOptimzers:
    def __init__(self):
        pass

    def Adam(
        self,
        params,
        lr=0.001,
        betas=(0.9, 0.999),
        eps=1e-08, 
        weight_decay=0, 
        amsgrad=False
    ):
        return optim.Adam(
            params, lr, betas, eps, weight_decay, amsgrad
        )