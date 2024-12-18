from torch.utils.data import Dataset
import os
from public.parseArgs import ParseArgs
from PIL import Image
from typing import List, Dict
import numpy as np

class ImageDataset(Dataset):
    def __init__(self, datas:List[Dict], datadir:str, folder:str, img_transformer, normalize=None):
        self.dataDict = datas
        self.normalize = normalize
        self.total = len(self.dataDict)
        self._image_transformer = img_transformer
        self.datadir = datadir
        self.folder = folder


    def get_image(self, index):
        filename = self.dataDict[index]["name"]
        ecfp6 = self.dataDict[index]["ecfp6"]
        imgPath = os.path.join(self.datadir, self.folder, filename)
        img = Image.open(imgPath).convert('RGB')
        ecfp6 = eval(ecfp6.replace("-", ","))
        ecfp6 = np.array(ecfp6, dtype=np.float32)
        return self._image_transformer(img), ecfp6

    def __getitem__(self, index):
        img, ecfp6 = self.get_image(index)
        if self.normalize is not None:
            img = self.normalize(img)
        return img, self.dataDict[index]["label"], self.dataDict[index].copy(), ecfp6


    def __len__(self):
        return self.total