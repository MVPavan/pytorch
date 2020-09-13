from imports import (
    torch,
    nn,
    F
)

class FCV1(nn.Module):
    def __init__(self,input_size,out_class_num):
        super(FCV1, self).__init__()

        self.fc1 = nn.Linear(input_size,50)
        self.fc2 = nn.Linear(50,out_class_num)

    def forward(self,x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x