import torch
from theloop import TheLoop
from dataset import GlassesDataset
from torch.nn import functional as F
from torchvision.datasets import MNIST, CIFAR10
import torchvision.models as models
import torchvision.transforms as transforms
from tqdm import tqdm, tqdm_notebook
from sklearn.metrics import accuracy_score
from argparse import ArgumentParser


def parse():
    parser = ArgumentParser()
    parser.add_argument("train_path")
    parser.add_argument("val_path")
    return parser.parse_args()


def batch_callback(state):
    model = state.model
    batch = state.batch
    device = state.device
    criterion = state.criterion

    out = model(batch["image"].to(device))
    state.loss = criterion(out, batch["label"].float().unsqueeze(1).to(device))


def val_callback(state):
    model = state.model
    dloader = state.data
    device = state.device
    
    predict = []
    ground_truth = []

    for batch in dloader:
        with torch.no_grad():
            out = F.sigmoid(model(batch["image"].to(device)).cpu())
            pred = (out > 0.5).int()

        predict += pred.tolist()
        ground_truth += batch["label"].tolist()

    accuracy = accuracy_score(predict, ground_truth)
    state.set_metric("accuracy", accuracy)



def get_model():
    model = torch.hub.load('pytorch/vision:v0.5.0', 'squeezenet1_0', pretrained=True)
    model.train()
    model.classifier[1] = torch.nn.Conv2d(512, 1, (1, 1))

    return model



def get_dataloaders(train_path, val_path):
    train_dataset = GlassesDataset(train_path)
    val_dataset = GlassesDataset(val_path, use_aug=False)

    return train_dataset, val_dataset


if __name__ == "__main__":
    model = get_model()
    theloop = TheLoop(model, 
                  "BCEWithLogitsLoss", batch_callback,
                  val_callback=val_callback,
                  optimizer_params={"lr": 1e-4},
                  logdir="./logdir",
                  val_rate=10,
                  device="cpu",
                  val_criterion_key="accuracy")

    args = parse()
    train_dataset, val_dataset = get_dataloaders(args.train_path, args.val_path)
    theloop.a(train_dataset, val_dataset, n_epoch=10)

