import os
from glob import glob
from random import shuffle
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor
import cv2
from albumentations import (
    HorizontalFlip, IAAPerspective, ShiftScaleRotate, CLAHE, RandomRotate90,
    Transpose, ShiftScaleRotate, Blur, OpticalDistortion, GridDistortion, HueSaturationValue,
    IAAAdditiveGaussianNoise, GaussNoise, MotionBlur, MedianBlur, RandomBrightnessContrast, IAAPiecewiseAffine,
    IAASharpen, IAAEmboss, Flip, OneOf, Compose
)


class GlassesDataset(Dataset):
    def __init__(self, datapath, use_aug=True):
        self.files_path = []
        self.totensor = ToTensor()
        self.use_aug = use_aug
        self.aug = Compose([RandomRotate90(),
                            Flip(),
                            Transpose(),
                            OneOf([
                                IAAAdditiveGaussianNoise(),
                                GaussNoise(),
                            ], p=0.2),
                            OneOf([
                                MotionBlur(p=.2),
                                MedianBlur(blur_limit=3, p=0.1),
                                Blur(blur_limit=3, p=0.1),
                            ], p=0.2),
                            ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=45, p=0.2),
                            OneOf([
                                OpticalDistortion(p=0.3),
                                GridDistortion(p=.1),
                                IAAPiecewiseAffine(p=0.3),
                            ], p=0.2),
                            OneOf([
                                CLAHE(clip_limit=2),
                                IAASharpen(),
                                IAAEmboss(),
                                RandomBrightnessContrast(),            
                            ], p=0.3),
                            HueSaturationValue(p=0.3),
                        ], p=0.8)

        for file_path in glob(os.path.join(datapath, "glasses/*.jpg")):
            self.files_path.append((file_path, 1))

        for file_path in glob(os.path.join(datapath, "no_glasses/*.jpg")):
            self.files_path.append((file_path, 0))

        shuffle(self.files_path)

    def __len__(self):
        return len(self.files_path)


    def __getitem__(self, indx):
        file_path, label = self.files_path[indx]
        image = cv2.imread(file_path)[:,:,(2,1,0)]
        if self.use_aug:   
            image = self.aug(image=image)["image"]
        image = cv2.resize(image, (256, 256))

        image = self.totensor(image)
        return {"image": image, "label": label}

        

