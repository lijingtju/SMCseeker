import pandas as pd
import os
from public.parseArgs import ParseArgs
from tools.tools import evaluateRule
import random
import shutil

class SplitDataset():
    def __init__(self, args:ParseArgs) -> None:
        self.args = args
        dataPathcsv = os.path.join(self.args.datadir, self.args.dataset, "{}.csv".format(self.args.dataset))
        dataPathxlsx = os.path.join(self.args.datadir, self.args.dataset, "{}.xlsx".format(self.args.dataset))
        if os.path.exists(dataPathcsv):
            dfData = pd.read_csv(dataPathcsv)
        elif os.path.exists(dataPathxlsx):
            dfData = pd.read_excel(dataPathxlsx)
        else:
            raise ValueError("Data does not exist:{}".format(os.path.join(self.args.datadir, self.args.dataset, "{}".format(self.args.dataset))))
        self.dataset = dfData.to_dict(orient='records')
        self.trainDataset = []
        self.valDataset = []
        self.testDataset = []

    def splitListByRatio(self, lst, x, y, z):
        total = x + y + z
        len_x = int(len(lst) * x // total)
        len_y = int(len(lst) * y // total)
        len_z = int(len(lst) - len_x - len_y)  
        part_x = lst[:len_x]
        part_y = lst[len_x:len_x+len_y]
        part_z = lst[len_x+len_y:]
        return part_x, part_y, part_z
    
    def save(self, dataset, name):
        source_dir = os.path.join(self.args.datadir, self.args.dataset, "raw")
        dst_dir = os.path.join(self.args.split_save_dir, self.args.dataset, name)
        if not os.path.exists(dst_dir):
            os.makedirs(dst_dir)

        for index, item in enumerate(dataset):
            source_path = os.path.join(source_dir, item["name"])
            dst_path = os.path.join(dst_dir, item["name"])
            shutil.copy(source_path, dst_path)
            print("\rcopy {}: {}/{}".format(name, index+1, len(dataset)), end='')
        print("")
        
        df = pd.DataFrame(dataset)
        df.to_csv(os.path.join(self.args.split_save_dir, self.args.dataset, "{}_{}.csv".format(self.args.dataset, name)))

    
    def run(self):
        positive = []
        negative = []
        for index, item in enumerate(self.dataset):
            if self.args.split_rule is None or evaluateRule(self.args.split_rule, item):
                positive.append(item)
            else:
                negative.append(item)
            print("\rapple rule: {}/{}".format(index+1, len(self.dataset)), end='')
        print("")

        if len(positive) == 0 and len(negative) == 0:
            return
        if len(positive):
            random.shuffle(positive)
            self.trainDataset, self.valDataset, self.testDataset = self.splitListByRatio(positive, self.args.split_scale[0], self.args.split_scale[1], self.args.split_scale[2])
        if len(negative):
            random.shuffle(negative)
            x,y,z = self.splitListByRatio(negative, self.args.split_scale[0], self.args.split_scale[1], self.args.split_scale[2])
            if len(x):
                self.trainDataset.extend(x)
            if len(y):
                self.valDataset.extend(y)
            if len(z):
                self.testDataset.extend(z)
        
        if len(self.trainDataset):
            random.shuffle(self.trainDataset)
            self.save(self.trainDataset, "train")
        
        if len(self.valDataset):
            random.shuffle(self.valDataset)
            self.save(self.valDataset, "val")

        if len(self.testDataset):
            random.shuffle(self.testDataset)
            self.save(self.testDataset, "test")
        print(len(self.trainDataset), len(self.valDataset), len(self.testDataset))