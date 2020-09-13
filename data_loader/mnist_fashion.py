from imports import DataLoader
from trainers.mnist_fashion import hparams
from imports import datasets, transforms

train_datset = datasets.FashionMNIST(
    root= "dataset/FashionMNIST/",
    train=True,
    transform=transforms.ToTensor(),
    download=True
)
test_datset = datasets.FashionMNIST(
    root= "dataset/FashionMNIST/",
    train=False,
    transform=transforms.ToTensor(),
    download=True
)

train_loader = DataLoader(
    dataset=train_datset,
    batch_size=hparams.batch,
    shuffle=True
)
test_loader = DataLoader(
    dataset=test_datset,
    batch_size=hparams.batch,
    shuffle=True
)