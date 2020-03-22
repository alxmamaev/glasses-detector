import os
import cv2
from glob import glob
from argparse import ArgumentParser
import torch
from torchvision.transforms import ToTensor

totensor = ToTensor()

def get_model(device):
    model = torch.hub.load('pytorch/vision:v0.5.0', 'squeezenet1_0', pretrained=False)
    model.classifier[1] = torch.nn.Conv2d(512, 1, (1, 1))
    model.load_state_dict(torch.load("model.pth", map_location="cpu"))
    model = model.to(device)

    return model

def parse():
    parser = ArgumentParser()
    parser.add_argument("path")
    return parser.parse_args()

def loadimg(imgpath, device):
    image = cv2.imread(imgpath)[:,:,(2,1,0)]
    image = cv2.resize(image, (256, 256))
    image = totensor(image)

    image.to(device)
    return image



def process_photo(photo_path, model, device):
    image = loadimg(photo_path, device).unsqueeze(0)
    output = torch.sigmoid(model(image)).cpu().squeeze().item()

    return output > 0.5




if __name__ == "__main__":
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
    
    model = get_model(device).eval()
    args = parse()

    photos_query = os.path.join(args.path, "*.jpg")
    for path in glob(photos_query):
        result = process_photo(path, model, device)
        if result:
            print(path)

